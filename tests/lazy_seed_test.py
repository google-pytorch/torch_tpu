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

"""Tests seeding the TPU generators before the TPU is initialized.

Like `torch.cuda`, seeding does not initialize the TPU; the seeds are applied
when the generators are created. TPU initialization happens at most once per
process, so each test seeds in a freshly spawned worker. This process must not
use the TPU, so that the worker can acquire it.
"""

import os
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.multiprocessing as mp
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils

_SEED_FNS: dict[str, Callable[[int], object]] = {
    "torch.manual_seed": torch.manual_seed,
    "torch.tpu.manual_seed": lambda seed: torch.tpu.manual_seed(seed),
    "torch.tpu.manual_seed_all": lambda seed: torch.tpu.manual_seed_all(seed),
}

# Sequences of (seed function name, seed) calls.
_SEED_CALLS: dict[str, list[tuple[str, int]]] = {
    "manual_seed": [("torch.tpu.manual_seed", 1234)],
    "manual_seed_all": [("torch.tpu.manual_seed_all", 1234)],
    "torch_manual_seed": [("torch.manual_seed", 1234)],
    # The later call wins, in either order.
    "manual_seed_all_then_manual_seed": [
        ("torch.tpu.manual_seed_all", 1),
        ("torch.tpu.manual_seed", 2),
    ],
    "manual_seed_then_manual_seed_all": [
        ("torch.tpu.manual_seed", 2),
        ("torch.tpu.manual_seed_all", 1),
    ],
}


def _seed(calls_name: str) -> None:
  for seed_fn_name, seed in _SEED_CALLS[calls_name]:
    _SEED_FNS[seed_fn_name](seed)


def _sample_after_seeding_before_and_after_init(
    local_rank: int, calls_name: str, output_path: str
) -> None:
  """Samples after seeding before TPU initialization, then after it."""
  del local_rank  # Unused.
  multiprocessing.parse_absl_flags()

  _seed(calls_name)
  if torch.tpu.is_initialized():
    raise AssertionError("seeding initialized the TPU")
  # This initializes the TPU and creates the generator with the seed.
  seeded_before_init = torch.rand(8, device="tpu").cpu()

  _seed(calls_name)
  seeded_after_init = torch.rand(8, device="tpu").cpu()

  torch.save(
      {
          "seeded_before_init": seeded_before_init,
          "seeded_after_init": seeded_after_init,
      },
      output_path,
  )


class LazySeedTest(seed_test_utils.MultiProcessRepeatableTest):

  @parameterized.named_parameters((name, name) for name in _SEED_CALLS)
  def test_seeding_before_init_matches_seeding_after_init(
      self, calls_name: str
  ) -> None:
    output_path = os.path.join(self.create_tempdir().full_path, "samples.pt")

    mp.spawn(
        _sample_after_seeding_before_and_after_init,
        args=(calls_name, output_path),
        nprocs=1,
        join=True,
    )

    samples = torch.load(output_path)
    seeded_before_init = samples["seeded_before_init"]
    seeded_after_init = samples["seeded_after_init"]
    self.assertTrue(
        torch.equal(seeded_before_init, seeded_after_init),
        f"seeding before TPU initialization sampled {seeded_before_init}, but"
        f" seeding after it sampled {seeded_after_init}",
    )


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
