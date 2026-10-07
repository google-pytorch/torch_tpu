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

"""Real-device tests for distributed_utils module.

Run on an 8-chip TPU host, where `dist_run` bootstraps the real TPU launch
environment and the workers use the `tpu_dist` backend. Every chip must be
driven by exactly one rank, so all 8 are used. Only behaviour that mocks cannot
reproduce lives here: tearing down and re-spawning real ranks on fresh
SliceBuilder ports, and the launch-environment caching across back-to-back
`dist_run` calls. The CPU tests live in `distributed_utils_test.py`.
"""

import os

from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.utils import hardware
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

_TPU_NPROC = 8
_ADDRESSES_ENV = "TORCH_TPU_SLICEBUILDER_ADDRESSES"
_ATTEMPTS_FILE = "attempts.txt"
_FAILED_ONCE_MARKER = "failed_once"

# Error text libtpu emits when the local TPU agent socket is unreachable, which
# `dist_run` treats as transient.
_TPU_AGENT_FAILURE = (
    "DEADLINE_EXCEEDED: failed to retrieve the PjRt client: TPU initialization"
    " failed: Failed to connect to unix:///var/run/tpu-plugin/vbar.sock"
)


def _record_attempt(output_dir: str) -> None:
  """Appends the SliceBuilder addresses this attempt ran with (rank 0 only)."""
  if int(os.environ["RANK"]) == 0:
    with open(os.path.join(output_dir, _ATTEMPTS_FILE), "a") as f:
      f.write(os.environ[_ADDRESSES_ENV] + "\n")


def _all_reduce_on_tpu() -> None:
  """Rendezvous via MASTER_PORT and run one collective on the real devices."""
  torch.distributed.init_process_group(backend="tpu_dist", init_method="env://")
  try:
    x = torch.ones(4, device="tpu")
    torch.distributed.all_reduce(x)
    world_size = torch.distributed.get_world_size()
    expected = torch.full((4,), float(world_size))
    assert torch.equal(x.cpu(), expected), f"expected {expected}, got {x}"
  finally:
    torch.distributed.destroy_process_group()


def _worker_record_and_all_reduce(output_dir: str) -> None:
  _record_attempt(output_dir)
  _all_reduce_on_tpu()


def _worker_fail_once_then_all_reduce(output_dir: str) -> None:
  """Rank 0 mimics a TPU-agent connect failure on the first attempt only."""
  _record_attempt(output_dir)
  marker = os.path.join(output_dir, _FAILED_ONCE_MARKER)
  if int(os.environ["RANK"]) == 0 and not os.path.exists(marker):
    with open(marker, "w"):
      pass
    raise RuntimeError(_TPU_AGENT_FAILURE)
  _all_reduce_on_tpu()


class DistributedUtilsTpuTest(seed_test_utils.MultiProcessRepeatableTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    device_count = hardware.get_tpu_device_count()
    if device_count < _TPU_NPROC:
      raise AssertionError(
          f"expected at least {_TPU_NPROC} TPU devices, got {device_count}"
      )

  def _read_attempts(self, output_dir: str) -> list[str]:
    with open(os.path.join(output_dir, _ATTEMPTS_FILE)) as f:
      return [line.strip() for line in f if line.strip()]

  def _local_ports(self, addresses: str) -> set[int]:
    """Returns the ports of a single-host `host:port,...` address list."""
    parts = addresses.split(",")
    self.assertLen(parts, _TPU_NPROC)
    ports = set()
    for address in parts:
      host, port = address.rsplit(":", 1)
      self.assertEqual(host, "localhost")
      ports.add(int(port))
    self.assertLen(ports, _TPU_NPROC)  # No duplicates within one attempt.
    return ports

  def _assert_disjoint_fresh_ports(self, attempts: list[str]) -> None:
    """Asserts every attempt used `_TPU_NPROC` ports unseen by any other."""
    self.assertLen(attempts, 2)
    first, second = (self._local_ports(a) for a in attempts)
    self.assertEmpty(
        first & second, f"Ports reused across attempts: {attempts}"
    )

  def test_dist_run_retries_on_tpu_agent_connect_failure(self) -> None:
    """Verifies a transient TPU-agent failure is retried on fresh ports."""
    output_dir = self.create_tempdir().full_path

    distributed_utils.dist_run(
        _TPU_NPROC,
        _worker_fail_once_then_all_reduce,
        output_dir,
        max_retries=3,
    )

    self._assert_disjoint_fresh_ports(self._read_attempts(output_dir))

  def test_dist_run_picks_fresh_slicebuilder_ports_per_attempt(self) -> None:
    """Verifies back-to-back dist_run calls each get their own ports."""
    output_dir = self.create_tempdir().full_path

    for _ in range(2):
      distributed_utils.dist_run(
          _TPU_NPROC, _worker_record_and_all_reduce, output_dir
      )

    attempts = self._read_attempts(output_dir)
    self._assert_disjoint_fresh_ports(attempts)
    # The parent process is left pointing at the ports of the last call.
    self.assertEqual(os.environ[_ADDRESSES_ENV], attempts[1])


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
