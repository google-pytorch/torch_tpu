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

"""Tests that the compile cache keeps apart programs of different world sizes.

A program without collectives is compiled for the default device assignment:
this process's device at world size 1, all devices of the job otherwise.

This matters when runs of different world sizes share one on-disk compile cache
(`TORCHINDUCTOR_CACHE_DIR`, by default shared by all processes of a user on a
host), e.g. a world-size-1 baseline followed by a tensor-parallel run of the
same model. A world-size-N run must not load the executable that the
world-size-1 run cached for the same graph: only the rank owning that one
device could run it, and every other rank would fail with "attempted to execute
with 1 argument lists when local device count is 0 (total replica count: 1,
partition count: 1)" and then hang.
"""

import json
import os
import threading

from absl.testing import absltest
import torch
from torch import distributed as dist
from torch._dynamo.utils import counters
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

# A rank that cannot run its executable hangs instead of raising; report it as
# failed after this long.
_WATCHDOG_S = 300


def _no_collectives(x: torch.Tensor) -> torch.Tensor:
  return torch.sin(x) * 2.0 + 1.0


def _input(device: str) -> torch.Tensor:
  return torch.arange(1024, dtype=torch.float32, device=device) / 1024.0


def run_no_collectives(cache_dir: str, out_dir: str, phase: str) -> None:
  """Compiles and runs `_no_collectives`; writes this rank's outcome.

  Args:
    cache_dir: The compile cache, shared by all ranks and phases.
    out_dir: Where each rank writes `<phase>_<rank>.json` with its error, if
      any, and its AOTAutograd cache counters.
    phase: The name of this run.
  """
  rank = int(os.environ.get("RANK", "0"))
  world_size = int(os.environ.get("WORLD_SIZE", "1"))
  path = os.path.join(out_dir, f"{phase}_{rank}.json")

  def write(error: str | None) -> None:
    with open(path, "w") as f:
      json.dump({"error": error, "aot": dict(counters["aot_autograd"])}, f)

  def on_timeout() -> None:
    write(f"hung for {_WATCHDOG_S} s")
    os._exit(1)  # pylint: disable=protected-access

  watchdog = threading.Timer(_WATCHDOG_S, on_timeout)
  watchdog.daemon = True
  watchdog.start()
  if world_size > 1:
    dist.init_process_group("tpu_dist")
  try:
    compiled = torch.compile(_no_collectives, backend="tpu", dynamic=False)
    with temporary_cache_dir(cache_dir):
      out = compiled(_input("tpu")).cpu()
    utils.assert_close(out, _no_collectives(_input("cpu")))
    write(None)
  except Exception as e:  # pylint: disable=broad-exception-caught
    write(f"{type(e).__name__}: {e}")
  finally:
    watchdog.cancel()
    if world_size > 1:
      dist.destroy_process_group()


class CompileCacheWorldSizeTest(seed_test_utils.MultiProcessRepeatableTest):

  def setUp(self):
    super().setUp()
    self._cache_dir = self.create_tempdir().full_path
    self._out_dir = self.create_tempdir().full_path
    # Use all host TPU chips: 8-chip v5e hosts cannot form a 4-chip sub-slice.
    self._world_size = distributed_utils.get_tpu_device_count()
    self.assertGreater(self._world_size, 1)

  def _run(self, phase: str, world_size: int) -> list[dict[str, int]]:
    """Runs `run_no_collectives` on `world_size` ranks; returns their counters."""
    distributed_utils.dist_run(
        world_size, run_no_collectives, self._cache_dir, self._out_dir, phase
    )
    outcomes = []
    for rank in range(world_size):
      with open(os.path.join(self._out_dir, f"{phase}_{rank}.json")) as f:
        outcomes.append(json.load(f))
    errors = {r: o["error"] for r, o in enumerate(outcomes) if o["error"]}
    self.assertEmpty(errors, f"failing ranks of {phase}")
    return [o["aot"] for o in outcomes]

  def test_world_size_1_entry_is_not_loaded_at_larger_world_size(self):
    (aot,) = self._run("world_size_1", 1)
    # Otherwise the multi-device run would not exercise the cache.
    self.assertEqual(aot.get("autograd_cache_saved"), 1, aot)
    # Ranks >= 1 fail if they load the world-size-1 executable.
    self._run(f"world_size_{self._world_size}", self._world_size)

  def test_entry_is_loaded_at_the_same_world_size(self):
    self._run("first", self._world_size)
    for aot in self._run("second", self._world_size):
      self.assertEqual(aot.get("autograd_cache_hit"), 1, aot)


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
