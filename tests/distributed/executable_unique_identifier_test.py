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

"""Tests for the unique identifier of executables."""

from collections.abc import Callable, Sequence
import pickle
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import distributed as dist
from torch_tpu._internal.compile import torch_tpu_compiled_executable
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests.distributed import continuous_test_utils

_WORLD = 8
# Layouts: each splits the world into groups of ranks.
_NEIGHBOR_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
_STRAIGHT_PAIRS = ((0, 2), (1, 3), (4, 6), (5, 7))

_Layout = Sequence[Sequence[int]]


def _my_group(layout: _Layout) -> dist.ProcessGroup:
  """Creates every group of `layout` and returns the one holding this rank."""
  mine = None
  for ranks in layout:
    group = dist.new_group(ranks=list(ranks))
    if dist.get_rank() in ranks:
      mine = group
  return mine


def _run_compiled(
    fn: Callable[..., Any], *inputs: torch.Tensor
) -> tuple[Any, torch_tpu_compiled_executable.CompiledArtifact]:
  """Compiles and runs `fn`, and returns its results and its executable."""
  debugs = []
  compiled = torch.compile(
      fn,
      backend="tpu",
      fullgraph=True,
      # Do not read the executable from a cache.
      options={"serializable": False, "debug_callback": debugs.append},
  )
  results = compiled(*inputs)
  (debug,) = debugs
  (executable,) = debug.compiled_executables
  return results, executable


def _compile_and_run(
    layouts: Sequence[_Layout],
) -> torch_tpu_compiled_executable.CompiledArtifact:
  """Runs one all_reduce per layout and returns the executable."""
  rank = dist.get_rank()
  groups = [_my_group(layout) for layout in layouts]

  @torch.tpu.distributed.spmd_safe
  def forward(x: torch.Tensor) -> torch.Tensor:
    for group in groups:
      dist.all_reduce(x, group=group)
    return x

  res, executable = _run_compiled(
      forward, torch.full((4,), rank + 1.0, device="tpu")
  )

  values = [r + 1.0 for r in range(_WORLD)]
  for layout in layouts:
    values = [
        sum(values[member] for member in group)
        for r in range(_WORLD)
        for group in layout
        if r in group
    ]
  utils.assert_close(res.cpu(), torch.full((4,), values[rank]))
  return executable


def _identifier_of_every_rank(key: str, identifier: str) -> list[str]:
  """Returns the `identifier` that each rank passes, indexed by rank."""
  store = dist.distributed_c10d._get_default_store()
  store.set(f"{key}/{dist.get_rank()}", identifier)
  return [store.get(f"{key}/{rank}").decode() for rank in range(_WORLD)]


def run_ranks_of_one_group_have_the_same_identifier(layout: _Layout) -> None:
  executable = _compile_and_run([layout])
  identifiers = _identifier_of_every_rank(
      f"one_group/{layout}", executable.executable_unique_identifier()
  )
  for group in layout:
    assert len({identifiers[rank] for rank in group}) == 1, identifiers


def run_ranks_of_a_tp_and_dp_program_have_the_same_identifier() -> None:
  rank = dist.get_rank()
  tp_layout = _NEIGHBOR_PAIRS
  dp_layout = ((0, 2, 4, 6), (1, 3, 5, 7))
  tp_group = _my_group(tp_layout)
  dp_group = _my_group(dp_layout)

  @torch.tpu.distributed.spmd_safe
  def step(
      activations: torch.Tensor, gradients: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    dist.all_reduce(activations, group=tp_group)
    dist.all_reduce(gradients, group=dp_group)
    return activations, gradients

  (activations, gradients), executable = _run_compiled(
      step,
      torch.full((4,), rank + 1.0, device="tpu"),
      torch.full((4,), rank + 1.0, device="tpu"),
  )
  for result, layout in ((activations, tp_layout), (gradients, dp_layout)):
    (group,) = [group for group in layout if rank in group]
    utils.assert_close(
        result.cpu(), torch.full((4,), sum(member + 1.0 for member in group))
    )

  identifiers = _identifier_of_every_rank(
      "tp_and_dp", executable.executable_unique_identifier()
  )
  assert len(set(identifiers)) == 1, identifiers


def run_order_of_layouts_changes_the_identifier() -> None:
  crossed_pairs = ((0, 3), (1, 2), (4, 7), (5, 6))
  straight_first = _compile_and_run([_STRAIGHT_PAIRS, crossed_pairs])
  crossed_first = _compile_and_run([crossed_pairs, _STRAIGHT_PAIRS])
  # The LLO fingerprints can be the same. The device assignments are not.
  assert (
      straight_first.executable_unique_identifier()
      != crossed_first.executable_unique_identifier()
  )


def run_layout_of_a_group_changes_the_identifier() -> None:
  # Two layouts that have the pair (0, 1).
  other_pairs = ((0, 1), (2, 4), (3, 6), (5, 7))
  world = (tuple(range(_WORLD)),)
  of_neighbor_pairs = _compile_and_run([_NEIGHBOR_PAIRS, world])
  of_other_pairs = _compile_and_run([other_pairs, world])
  assert (
      of_neighbor_pairs.executable_unique_identifier()
      != of_other_pairs.executable_unique_identifier()
  )


def run_reloaded_executable_has_the_same_identifier() -> None:
  (executable,) = _compile_and_run([_NEIGHBOR_PAIRS]).compiled_executables
  reloaded = pickle.loads(pickle.dumps(executable))
  assert (
      reloaded.executable_unique_identifier()
      == executable.executable_unique_identifier()
  )


class ExecutableUniqueIdentifierTest(
    continuous_test_utils.TpuMultiProcContinuousTest
):
  world_size = _WORLD

  @parameterized.named_parameters(
      ("neighbor_pairs", _NEIGHBOR_PAIRS),
      ("straight_pairs", _STRAIGHT_PAIRS),
  )
  def test_ranks_of_one_group_have_the_same_identifier(
      self, layout: _Layout
  ) -> None:
    run_ranks_of_one_group_have_the_same_identifier(layout)

  def test_ranks_of_a_tp_and_dp_program_have_the_same_identifier(self) -> None:
    run_ranks_of_a_tp_and_dp_program_have_the_same_identifier()

  def test_order_of_layouts_changes_the_identifier(self) -> None:
    run_order_of_layouts_changes_the_identifier()

  def test_layout_of_a_group_changes_the_identifier(self) -> None:
    run_layout_of_a_group_changes_the_identifier()

  def test_reloaded_executable_has_the_same_identifier(self) -> None:
    run_reloaded_executable_has_the_same_identifier()


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
