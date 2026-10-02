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

"""Tests the unified `distributed` module for single-host and multi-host."""

# pylint: disable=protected-access

import contextlib
import io
import os
import string
import subprocess
import sys
from typing import Any
from unittest import mock

from absl.testing import absltest
from torch.tpu import distributed  # pyrefly: ignore[missing-import]
from torch_tpu._internal.distributed import distributed as internal_distributed
from torch_tpu._internal.utils import hardware
from tests import env_test_utils
from tests import seed_test_utils


def _mock_env(
    values: dict[str, str],
) -> contextlib.AbstractContextManager[None]:
  """Makes the environment look exactly like `values` to the launcher."""
  return env_test_utils.scoped_env(values, clear=True)


class _FakeCoordsChild:
  """Stands in for the child process that reports the local device coordinates.

  Like the real child, it fails unless it is launched as this module's
  coordinate printer with a `PYTHONPATH` from which TorchTPU can be imported.
  `$NAME` placeholders in its output are replaced with the values of the
  environment variables it was launched with. Its output is bytes unless
  `text=True`, and a failure raises only if `check=True`, as with
  `subprocess.run()`.
  """

  def __init__(self, stdout: str = "", returncode: int = 0):
    self._stdout = stdout
    self._returncode = returncode
    self.launches = 0

  def __call__(
      self,
      args: list[str],
      *,
      env: dict[str, str] | None = None,
      check: bool = False,
      text: bool = False,
      **unused_kwargs: Any,
  ) -> subprocess.CompletedProcess[Any]:
    """Runs the fake child process, with the signature of `subprocess.run`."""
    self.launches += 1
    python_path = (env or {}).get("PYTHONPATH", "").split(os.pathsep)
    can_import_torch_tpu = all(p in python_path for p in sys.path if p)
    is_coords_printer = args == [
        sys.executable,
        "-m",
        "torch_tpu._internal.distributed.distributed",
        "--print_local_device_coords",
    ]
    if is_coords_printer and can_import_torch_tpu:
      returncode = self._returncode
      stdout = string.Template(self._stdout).safe_substitute(env or {})
    else:
      returncode, stdout = 1, ""
    if check and returncode:
      raise subprocess.CalledProcessError(returncode, args, stdout)
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout if text else stdout.encode(),
        stderr="" if text else b"",
    )


def _coords_child(
    coords: list[int], prefix_lines: str = ""
) -> _FakeCoordsChild:
  """Returns a fake child process that reports `coords` on stdout."""
  return _FakeCoordsChild(
      stdout=f"{prefix_lines}TORCH_TPU_LOCAL_DEVICE_COORDS={coords}\n"
  )


class EnvironmentTest(seed_test_utils.RepeatableTest):

  def test_singlehost_auto_detection(self):
    with _mock_env({}):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2x2x1"),
          mock.patch.object(hardware, "get_tpu_device_count", return_value=4),
      ):
        distributed.set_tpu_launch_env()
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
        distributed.set_tpu_launch_env()
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
      distributed.set_tpu_launch_env(nproc_per_node=8)
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
      distributed.set_tpu_launch_env()

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
          internal_distributed._print_launch_env()
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
          internal_distributed._print_launch_env()
        self.assertIn('NODE_RANK="3"\n', buf.getvalue())

  def test_print_launch_env_multihost_with_coordinates(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "1",
    }):
      with (
          mock.patch.object(hardware, "get_tpu_device_count", return_value=4),
          mock.patch.object(subprocess, "run", _coords_child([2, 0, 0])),
      ):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
          internal_distributed._print_launch_env()
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
    self.assertEqual(internal_distributed._parse_topology("4x4x4"), (4, 4, 4))
    self.assertEqual(internal_distributed._parse_topology("2,2,1"), (2, 2, 1))
    self.assertEqual(internal_distributed._parse_topology("2,2,1,2"), (2, 2, 1))
    self.assertEqual(internal_distributed._parse_topology("2x4"), (2, 4, 1))
    self.assertEqual(internal_distributed._parse_topology("2X4"), (2, 4, 1))
    self.assertEqual(internal_distributed._parse_topology("4"), (4, 1, 1))

  def test_get_slice_host_coordinates_1d(self):
    # 1D topology "4" -> (4, 1, 1), with (2, 2, 1) chips_per_host:
    # Bx = 4 // 2 = 2, By = 1, Bz = 1 -> 2 hosts along X.
    hosts = internal_distributed._get_slice_host_coordinates("4")
    self.assertEqual(hosts, [(0, 0, 0), (1, 0, 0)])

  def test_get_slice_host_coordinates_2d(self):
    # 2D topology "2x4" -> (2, 4, 1), with (2, 2, 1) chips_per_host:
    # Bx = 1, By = 2, Bz = 1 -> 2 hosts along Y.
    hosts = internal_distributed._get_slice_host_coordinates("2x4")
    self.assertEqual(hosts, [(0, 0, 0), (0, 1, 0)])

  def test_get_slice_host_coordinates_3d(self):
    # 3D topology "4x4x4" -> (4, 4, 4), with (2, 2, 1) chips_per_host:
    # Bx = 2, By = 2, Bz = 4 -> 16 hosts.
    # Sorted reverse-lexicographically: Z-major, Y-intermediate, X-minor.
    hosts = internal_distributed._get_slice_host_coordinates("4x4x4")
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
    hosts = internal_distributed._get_slice_host_coordinates("2,2,1")
    self.assertEqual(hosts, [(0, 0, 0)])

  def test_get_slice_host_coordinates_4d(self):
    # 4D topology "4,4,4,2" -> parsed to (4, 4, 4), 16 hosts.
    hosts_4d = internal_distributed._get_slice_host_coordinates("4,4,4,2")
    hosts_3d = internal_distributed._get_slice_host_coordinates("4x4x4")
    self.assertEqual(hosts_4d, hosts_3d)

    # 4D topology "2,2,1,2" -> parsed to (2, 2, 1), 1 host.
    hosts_single = internal_distributed._get_slice_host_coordinates("2,2,1,2")
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
            internal_distributed._calculate_topological_node_rank(coords, topo),
            expected_rank,
        )

  def test_calculate_topological_node_rank_various_topologies(self):
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [0, 0, 0], "2,2,1"
        ),
        0,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [1, 1, 0], "2,2,1"
        ),
        0,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [2, 2, 0, 1], "4,4,4,2"
        ),
        3,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [0, 2, 0], "2x4x1"
        ),
        1,
    )
    # Missing trailing coordinates default to 0.
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank([2, 2], "4x4x4"),
        3,
    )

  def test_calculate_topological_node_rank_fallback(self):
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [], "4x4x4", fallback_index="2"
        ),
        2,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            None, "4x4x4", fallback_index=3
        ),
        3,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [0, 0, 0], "", fallback_index="1"
        ),
        1,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [0, 0, 0], None, fallback_index="1"
        ),
        1,
    )

  def test_calculate_topological_node_rank_out_of_bounds_fallback(self):
    # Chip coordinates out of bounds: host (5, 0, 0) does not exist in 4x4x4.
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [10, 0, 0], "4x4x4", fallback_index="7"
        ),
        7,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [10, 0, 0], "4x4x4", fallback_index=8
        ),
        8,
    )

    # Negative coordinates out of bounds
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [-1, 0, 0], "4x4x4", fallback_index="4"
        ),
        4,
    )

  def test_calculate_topological_node_rank_invalid_topology_fallback(self):
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
            [0, 0, 0], "4x4xfour", fallback_index="6"
        ),
        6,
    )
    self.assertEqual(
        internal_distributed._calculate_topological_node_rank(
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
      with mock.patch.object(subprocess, "run", _coords_child([2, 0, 0])):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")

  def test_set_tpu_launch_env_with_coordinates_and_no_job_completion_index(
      self,
  ):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
    }):
      with mock.patch.object(subprocess, "run", _coords_child([0, 2, 0])):
        distributed.set_tpu_launch_env()
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
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_empty_coordinates_fallback(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with mock.patch.object(subprocess, "run", _coords_child([])):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_out_of_bounds_coordinates_fallback(self):
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "7",
    }):
      with mock.patch.object(subprocess, "run", _coords_child([10, 0, 0])):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "7")

  def test_set_tpu_launch_env_without_tpu_devices_skips_query(self):
    # The child would report a host with rank 1, so rank 3 shows it never ran.
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with (
          mock.patch.object(hardware, "get_tpu_device_count", return_value=0),
          mock.patch.object(subprocess, "run", _coords_child([2, 0, 0])),
      ):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")

  def test_set_tpu_launch_env_without_topology_skips_query(self):
    child = _coords_child([2, 0, 0])
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "3",
    }):
      with mock.patch.object(subprocess, "run", child):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "3")
    # Without a slice topology, coordinates cannot be ranked, so spawning the
    # child would only cost time.
    self.assertEqual(child.launches, 0)

  def test_set_tpu_launch_env_ignores_auto_detected_topology(self):
    # The auto-detected topology describes this host alone, not the slice, so
    # it must not be used to rank hosts. Ranking the child's coordinates in it
    # would give rank 0.
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "JOB_COMPLETION_INDEX": "1",
    }):
      with (
          mock.patch.object(
              hardware, "get_tpu_topology", return_value="2,2,1,2"
          ),
          mock.patch.object(subprocess, "run", _coords_child([0, 0, 0])),
      ):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")
      self.assertEqual(os.environ.get("TORCH_TPU_TOPOLOGY"), "2,2,1,2")

  def test_set_tpu_launch_env_keeps_pinned_node_rank(self):
    # The child would report a host with rank 0.
    with _mock_env({
        "TPU_WORKER_HOSTNAMES": "host1,host2",
        "TORCH_TPU_TOPOLOGY": "4x4x4",
        "NODE_RANK": "1",
    }):
      with mock.patch.object(subprocess, "run", _coords_child([0, 0, 0])):
        distributed.set_tpu_launch_env()
      self.assertEqual(os.environ.get("NODE_RANK"), "1")

  def test_set_tpu_launch_env_singlehost_skips_query(self):
    child = _coords_child([0, 0, 0])
    with _mock_env({
        "TORCH_TPU_TOPOLOGY": "2,2,1",
    }):
      with (
          mock.patch.object(hardware, "get_tpu_topology", return_value="2,2,1"),
          mock.patch.object(subprocess, "run", child),
      ):
        distributed.set_tpu_launch_env()
      self.assertIsNone(os.environ.get("NODE_RANK"))
    # A single host needs no rank, so spawning the child would only cost time.
    self.assertEqual(child.launches, 0)


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
          mock.patch.object(subprocess, "run", _coords_child([1, 2, 3, 0])),
      ):
        self.assertIsNone(internal_distributed._query_local_device_coords())

  def test_child_process_inherits_environment(self):
    child = _FakeCoordsChild(
        stdout="TORCH_TPU_LOCAL_DEVICE_COORDS=$FAKE_COORDS"
    )
    with _mock_env({"FAKE_COORDS": "[1, 2, 3, 0]"}):
      with mock.patch.object(subprocess, "run", child):
        self.assertEqual(
            internal_distributed._query_local_device_coords(), [1, 2, 3, 0]
        )

  def test_child_exit_failure_returns_none(self):
    child = _FakeCoordsChild(
        stdout="TORCH_TPU_LOCAL_DEVICE_COORDS=[1, 2, 3, 0]\n", returncode=1
    )
    with _mock_env({}):
      with mock.patch.object(subprocess, "run", child):
        self.assertIsNone(internal_distributed._query_local_device_coords())

  def test_ignores_other_stdout_lines(self):
    with _mock_env({}):
      with mock.patch.object(
          subprocess,
          "run",
          _coords_child(
              [2, 0, 1], prefix_lines="Registered Python module for 'tpu'.\n"
          ),
      ):
        self.assertEqual(
            internal_distributed._query_local_device_coords(), [2, 0, 1]
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
            self.assertIsNone(internal_distributed._query_local_device_coords())

  def test_missing_coordinates_line_returns_none(self):
    with _mock_env({}):
      with mock.patch.object(
          subprocess, "run", _FakeCoordsChild(stdout="unrelated\n")
      ):
        self.assertIsNone(internal_distributed._query_local_device_coords())

  def test_malformed_coordinates_line_returns_none(self):
    with _mock_env({}):
      with mock.patch.object(
          subprocess,
          "run",
          _FakeCoordsChild(stdout="TORCH_TPU_LOCAL_DEVICE_COORDS=[1, 2\n"),
      ):
        self.assertIsNone(internal_distributed._query_local_device_coords())

  def test_print_local_device_coords(self):
    with mock.patch(
        "torch_tpu._internal.distributed.tpu_distributed.local_device_coords",
        return_value=[2, 0, 1],
    ):
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        internal_distributed._print_local_device_coords()
    self.assertEqual(
        buf.getvalue(), "TORCH_TPU_LOCAL_DEVICE_COORDS=[2, 0, 1]\n"
    )


if __name__ == "__main__":
  absltest.main()
