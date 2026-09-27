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

"""Tests multi-process multi-device RNG behaviors.

Tests TPU and CUDA backends in spawned child processes across devices.
"""

import os
import queue
from typing import Any, Callable

from absl import flags
from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.utils import hardware
from torch_tpu._internal.distributed import multiprocessing
from tests import accelerator_test_utils
from tests import seed_test_utils
from tests.distributed import distributed_utils

_BACKEND = flags.DEFINE_string(
    "backend", "tpu", "The backend to test: 'tpu' or 'gpu'."
)


# Worker callbacks are defined as standalone module-level functions rather
# than lambdas or closures so they can be serialized by multiprocessing's
# spawn context (ForkingPickler) across process boundaries.
def _set_different_seeds(backend_mod: Any, rank: int) -> None:
  """Sets manual seed to base 42 plus worker rank."""
  backend_mod.manual_seed(42 + rank)


def _get_initial_seed(backend_mod: Any, dev: torch.device, rank: int) -> int:
  """Retrieves accelerator random initial seed."""
  del backend_mod, dev, rank  # Unused.
  return torch.accelerator.random.initial_seed()


def _set_same_seed(backend_mod: Any, rank: int) -> None:
  """Sets manual seed to a fixed value of 42 across all ranks."""
  backend_mod.manual_seed(42)


def _gen_rand_tensors(
    backend_mod: Any, dev: torch.device, rank: int
) -> list[list[float]]:
  """Generates a stack of eager random tensors on dev as nested lists."""
  tensors = [torch.rand(10, device=dev).cpu().tolist() for _ in range(5)]
  return tensors


def _gen_compiled_rand(
    backend_mod: Any, dev: torch.device, rank: int
) -> list[float]:
  """Generates random output from a compiled function on dev as a list."""
  torch._inductor.config.compile_threads = 1

  def fn(x):
    return torch.rand_like(x) + torch.rand_like(x)

  compiled_fn = torch.compile(fn, fullgraph=True)
  x = torch.zeros(10, device=dev)
  return compiled_fn(x).cpu().tolist()


def _multiprocess_worker(
    results_queue: queue.Queue[tuple[int, Any]],
    backend_name: str,
    seed_fn: Callable[[Any, int], None] | None,
    ops_fn: Callable[[Any, torch.device, int], Any],
) -> None:
  """Worker function executing in a spawned child process.

  Retrieves device configuration based on LOCAL_RANK, invokes seed_fn to set
  the device RNG seed, executes ops_fn to produce test output, and places the
  resulting data into results_queue.

  Args:
    results_queue: Multiprocessing queue to send tuples of (rank, return value
      from ops_fn) back to the runner process.
    backend_name: Name of backend ('tpu' or 'gpu').
    seed_fn: Optional callable taking (backend_mod, rank) to initialize RNG
      seed.
    ops_fn: Callable taking (backend_mod, dev, rank) returning output result.
  """
  rank = int(os.environ["LOCAL_RANK"])
  backend_mod = accelerator_test_utils.get_backend_module(backend_name)
  # On GPU/CUDA, all devices are visible, so set to rank.
  # On TPU, only one device is visible, so set to 0.
  device_idx = rank if backend_name in ("gpu", "cuda") else 0
  dev = accelerator_test_utils.set_active_device(backend_name, device_idx)

  if seed_fn is not None:
    seed_fn(backend_mod, rank)
  out = ops_fn(backend_mod, dev, rank)
  results_queue.put((rank, out))


class MultiProcessTest(seed_test_utils.MultiProcessRepeatableTest):
  """Tests multi-process multi-device RNG determinism and isolation.

  Uses MultiProcessRepeatableTest (seed_in_setup = False) and
  distributed_utils.dist_run to spawn workers without initializing the
  device runtime in the parent runner process.
  """

  def setUp(self):
    super().setUp()
    self.backend = _BACKEND.value
    self.backend_mod = accelerator_test_utils.get_backend_module(self.backend)
    self.num_devices = self._get_num_devices()
    self.assertGreater(
        self.num_devices,
        1,
        "Test target must be configured with multiple devices to verify"
        " multi-process multi-device RNG operations.",
    )

  def _get_num_devices(self) -> int:
    if self.backend == "tpu":
      return hardware.get_tpu_device_count()
    return min(torch.cuda.device_count(), 2)

  def _run_distributed(
      self,
      seed_fn: Callable[[Any, int], None] | None,
      ops_fn: Callable[[Any, torch.device, int], Any],
      num_devices: int | None = None,
  ) -> list[Any]:
    if num_devices is None:
      num_devices = self.num_devices

    ctx = multiprocessing.get_context("spawn")
    results_queue = ctx.Queue()

    distributed_utils.dist_run(
        num_devices,
        _multiprocess_worker,
        results_queue,
        self.backend,
        seed_fn,
        ops_fn,
    )

    results = {}
    for _ in range(num_devices):
      rank, out = results_queue.get(timeout=60)
      results[rank] = out

    return [results[i] for i in range(num_devices)]

  def test_multiprocess_different_seeds_isolate_seed(self):
    """Verifies spawned processes have isolated RNG seeds."""
    seeds = self._run_distributed(
        _set_different_seeds,
        _get_initial_seed,
    )
    for i in range(self.num_devices):
      self.assertEqual(seeds[i], 42 + i)
    self.assertNotEqual(seeds[0], seeds[1])

  def test_multiprocess_eager_device_isolation_deterministic(self):
    """Verifies spawned processes per device produce deterministic outputs."""
    trial1 = self._run_distributed(
        _set_different_seeds,
        _gen_rand_tensors,
    )
    trial2 = self._run_distributed(
        _set_different_seeds,
        _gen_rand_tensors,
    )

    for res1, res2 in zip(trial1, trial2):
      self.assertEqual(res1, res2)
    self.assertNotEqual(trial1[0], trial1[1])

  def test_multiprocess_identical_seed_across_devices_matches(self):
    """Verifies identical seed across spawned processes matches output."""
    outputs = self._run_distributed(
        _set_same_seed,
        _gen_rand_tensors,
    )
    for i in range(1, len(outputs)):
      self.assertEqual(outputs[0], outputs[i])

  def test_multiprocess_compiled_device_deterministic(self):
    """Verifies compiled function across processes is deterministic."""
    trial1 = self._run_distributed(
        _set_different_seeds,
        _gen_compiled_rand,
    )
    trial2 = self._run_distributed(
        _set_different_seeds,
        _gen_compiled_rand,
    )

    for res1, res2 in zip(trial1, trial2):
      self.assertEqual(res1, res2)
    self.assertNotEqual(trial1[0], trial1[1])


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
