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

"""Unit tests for the fuse_collective_relayouts pass."""

import operator

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch.fx.experimental import proxy_tensor
from torch.testing._internal.distributed import fake_pg
from torch_tpu._internal.compile.fx_passes import fuse_collective_relayouts
from tests import seed_test_utils

fx = torch.fx

# pylint: disable=protected-access
_ALL_GATHER = torch.ops._c10d_functional.all_gather_into_tensor.default
_REDUCE_SCATTER = torch.ops._c10d_functional.reduce_scatter_tensor.default
_WAIT = torch.ops._c10d_functional.wait_tensor.default
_ALL_GATHER_DIM = torch.ops.tpu._all_gather_dim.default
_REDUCE_SCATTER_DIM = torch.ops.tpu._reduce_scatter_dim.default
# pylint: enable=protected-access

_WORLD_SIZE = 2
_GROUP = "group"


def _targets(gm: fx.GraphModule) -> list[object]:
  return [n.target for n in gm.graph.nodes if n.op == "call_function"]


def _find(gm: fx.GraphModule, target: object) -> list[fx.Node]:
  return [n for n in gm.graph.nodes if n.target == target]


def _with_val(node: fx.Node, *shape: int) -> fx.Node:
  node.meta["val"] = torch.empty(shape, device="meta")
  return node


def _hand_built_all_gather(
    chunk_order: list[int], num_chunks: int = _WORLD_SIZE
) -> fx.GraphModule:
  """Returns `cat(split(wait(all_gather(x)), dim=0)[chunk_order], dim=1)`."""
  graph = fx.Graph()
  x = _with_val(graph.placeholder("x"), 4, 6)
  all_gather = _with_val(
      graph.call_function(_ALL_GATHER, (x, _WORLD_SIZE, _GROUP)), 8, 6
  )
  wait = _with_val(graph.call_function(_WAIT, (all_gather,)), 8, 6)
  split = graph.call_function(
      torch.ops.aten.split.Tensor, (wait, 8 // num_chunks)
  )
  chunks = [
      graph.call_function(operator.getitem, (split, i))
      for i in range(num_chunks)
  ]
  cat = graph.call_function(
      torch.ops.aten.cat.default, ([chunks[i] for i in chunk_order], 1)
  )
  _with_val(cat, 4, 6 * len(chunk_order) * _WORLD_SIZE // num_chunks)
  graph.output(cat)
  return fx.GraphModule(torch.nn.Module(), graph)


def _hand_built_reduce_scatter(
    reduce_op: str = "sum", use_slices: bool = False
) -> fx.GraphModule:
  """Returns `wait(reduce_scatter(cat(chunk(y, dim=1), dim=0)))`."""
  graph = fx.Graph()
  y = _with_val(graph.placeholder("y"), 4, 6)
  if use_slices:
    # `y[:, :3]` and `y[:, 3:]`, as traced from Python slicing.
    chunks = [
        graph.call_function(torch.ops.aten.slice.Tensor, (y, 1, None, 3)),
        graph.call_function(
            torch.ops.aten.slice.Tensor, (y, 1, 3, 9223372036854775807)
        ),
    ]
  else:
    split = graph.call_function(torch.ops.aten.split.Tensor, (y, 3, 1))
    chunks = [
        graph.call_function(operator.getitem, (split, i))
        for i in range(_WORLD_SIZE)
    ]
  cat = _with_val(
      graph.call_function(torch.ops.aten.cat.default, (chunks,)), 8, 3
  )
  reduce_scatter = _with_val(
      graph.call_function(
          _REDUCE_SCATTER, (cat, reduce_op, _WORLD_SIZE, _GROUP)
      ),
      4,
      3,
  )
  wait = _with_val(graph.call_function(_WAIT, (reduce_scatter,)), 4, 3)
  graph.output(wait)
  return fx.GraphModule(torch.nn.Module(), graph)


def _reduce_scatter_on_dim_1(reduce_op: str):
  def fn(y):
    return funcol.reduce_scatter_tensor(y, reduce_op, 1, dist.group.WORLD)

  return fn


class FuseCollectiveRelayoutsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    dist.init_process_group(
        "fake", store=fake_pg.FakeStore(), rank=0, world_size=_WORLD_SIZE
    )
    self.addCleanup(dist.destroy_process_group)
    self.group_name = dist.group.WORLD.group_name

  def _trace(self, fn, *shapes: tuple[int, ...]) -> fx.GraphModule:
    """Traces `fn` to an ATen graph like the ones AOTAutograd produces."""
    args = [torch.empty(shape, device="meta") for shape in shapes]
    return proxy_tensor.make_fx(fn, tracing_mode="fake")(*args)

  def test_all_gather_on_dim_1_is_fused(self):
    gm = self._trace(
        lambda x: funcol.all_gather_tensor(x, 1, dist.group.WORLD), (4, 6)
    )
    self.assertIn(torch.ops.aten.cat.default, _targets(gm))

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), [_ALL_GATHER_DIM], gm.print_readable(False))
    (fused,) = _find(gm, _ALL_GATHER_DIM)
    x = gm.graph.find_nodes(op="placeholder")[0]
    self.assertEqual(fused.args, (x, 1, _WORLD_SIZE, self.group_name))
    self.assertEqual(fused.meta["val"].shape, (4, 12))

  def test_all_gather_on_negative_dim_is_fused_on_wrapped_dim(self):
    gm = self._trace(
        lambda x: funcol.all_gather_tensor(x, -1, dist.group.WORLD), (2, 4, 6)
    )

    fuse_collective_relayouts.apply(gm)

    (fused,) = _find(gm, _ALL_GATHER_DIM)
    self.assertEqual(fused.args[1:3], (2, _WORLD_SIZE))
    self.assertEqual(fused.meta["val"].shape, (2, 4, 12))
    self.assertEmpty(_find(gm, _ALL_GATHER))

  def test_reduce_scatter_on_dim_1_is_fused(self):
    for reduce_op in ("sum", "avg"):
      with self.subTest(reduce_op=reduce_op):
        gm = self._trace(_reduce_scatter_on_dim_1(reduce_op), (4, 6))
        self.assertIn(torch.ops.aten.cat.default, _targets(gm))

        fuse_collective_relayouts.apply(gm)

        self.assertEqual(
            _targets(gm), [_REDUCE_SCATTER_DIM], gm.print_readable(False)
        )
        (fused,) = _find(gm, _REDUCE_SCATTER_DIM)
        y = gm.graph.find_nodes(op="placeholder")[0]
        self.assertEqual(
            fused.args, (y, reduce_op, 1, _WORLD_SIZE, self.group_name)
        )
        self.assertEqual(fused.meta["val"].shape, (4, 3))

  def test_round_trip_is_fused(self):
    def fn(x):
      gathered = funcol.all_gather_tensor(x, 1, dist.group.WORLD)
      return funcol.reduce_scatter_tensor(
          gathered * 2, "sum", 1, dist.group.WORLD
      )

    gm = self._trace(fn, (4, 6))

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(
        _targets(gm),
        [_ALL_GATHER_DIM, torch.ops.aten.mul.Tensor, _REDUCE_SCATTER_DIM],
        gm.print_readable(False),
    )

  def test_graph_module_is_recompiled(self):
    gm = self._trace(
        lambda x: funcol.all_gather_tensor(x, 1, dist.group.WORLD), (4, 6)
    )

    fuse_collective_relayouts.apply(gm)

    self.assertIn("_all_gather_dim", gm.code)
    self.assertNotIn("all_gather_into_tensor", gm.code)

  def test_independent_patterns_are_all_fused(self):
    def fn(x, y):
      return (
          funcol.all_gather_tensor(x, 1, dist.group.WORLD),
          funcol.all_gather_tensor(y, 2, dist.group.WORLD),
          funcol.reduce_scatter_tensor(x, "sum", 1, dist.group.WORLD),
      )

    gm = self._trace(fn, (4, 6), (2, 4, 6))

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(
        _targets(gm),
        [_ALL_GATHER_DIM, _ALL_GATHER_DIM, _REDUCE_SCATTER_DIM],
        gm.print_readable(False),
    )

  def test_reduce_scatter_with_slice_chunks_is_fused(self):
    gm = _hand_built_reduce_scatter(use_slices=True)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), [_REDUCE_SCATTER_DIM])
    (fused,) = _find(gm, _REDUCE_SCATTER_DIM)
    self.assertEqual(fused.args[1:], ("sum", 1, _WORLD_SIZE, _GROUP))

  def test_all_gather_on_dim_0_is_unchanged(self):
    gm = self._trace(
        lambda x: funcol.all_gather_tensor(x, 0, dist.group.WORLD), (4, 6)
    )
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)
    self.assertEmpty(_find(gm, _ALL_GATHER_DIM))

  def test_all_gather_with_other_users_is_unchanged(self):
    # The dim-0 result is also returned, so the original all-gather has to stay
    # and fusing would add a second collective.
    def fn(x):
      gathered = funcol.all_gather_tensor(x, 0, dist.group.WORLD)
      return gathered, torch.cat(torch.chunk(gathered, _WORLD_SIZE), dim=1)

    gm = self._trace(fn, (4, 6))
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)

  def test_reduce_scatter_with_other_users_of_cat_is_unchanged(self):
    def fn(y):
      relayout = torch.cat(torch.chunk(y, _WORLD_SIZE, dim=1))
      return relayout, funcol.reduce_scatter_tensor(
          relayout, "sum", 0, dist.group.WORLD
      )

    gm = self._trace(fn, (4, 6))
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)

  def test_reduce_scatter_on_dim_0_is_unchanged(self):
    gm = self._trace(
        lambda y: funcol.reduce_scatter_tensor(y, "sum", 0, dist.group.WORLD),
        (4, 6),
    )
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)

  def test_out_of_order_chunks_are_unchanged(self):
    gm = _hand_built_all_gather(chunk_order=[1, 0])
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)

  def test_chunks_not_matching_group_size_are_unchanged(self):
    gm = _hand_built_all_gather(chunk_order=[0, 1, 2, 3], num_chunks=4)
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)

  def test_hand_built_all_gather_is_fused(self):
    gm = _hand_built_all_gather(chunk_order=[0, 1])

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), [_ALL_GATHER_DIM])

  def test_unsupported_reduce_op_is_unchanged(self):
    gm = _hand_built_reduce_scatter(reduce_op="premul_sum")
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)

  def test_missing_shape_metadata_is_unchanged(self):
    gm = _hand_built_reduce_scatter()
    for node in gm.graph.nodes:
      node.meta.pop("val", None)
    before = _targets(gm)

    fuse_collective_relayouts.apply(gm)

    self.assertEqual(_targets(gm), before)


if __name__ == "__main__":
  absltest.main()
