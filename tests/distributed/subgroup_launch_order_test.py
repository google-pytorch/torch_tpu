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

"""Tests that subgroup collectives run independently of the other subgroups.

A program whose collectives all run over one proper subgroup of the slice is
compiled for that subgroup's devices, so the members of one subgroup may issue
any number of such programs without the other subgroups launching anything.
Pipeline stages with tensor parallelism inside each stage rely on this: every
stage runs its own sequence of stage-local collectives while all stages join
only the hand-off collective. A multi-device kernel tagged with the subgroup's
devices (JaxCallable's collective_devices) is restricted the same way.
"""

import os
import signal

from absl import logging
from absl.testing import absltest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch_tpu._internal.distributed.launchers import singlehost_wrapper
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import seed_test_utils
from tests.distributed import distributed_utils

_WORLD = 8
_PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]]
_COLUMNS = [[0, 2, 4, 6], [1, 3, 5, 7]]


def _make_groups(rank):
  """Creates every pair and column group on every rank; returns this rank's."""
  my_pair, my_pair_ranks = None, None
  for ranks in _PAIRS:
    group = dist.new_group(ranks=ranks)
    if rank in ranks:
      my_pair, my_pair_ranks = group, ranks
  my_column = None
  for ranks in _COLUMNS:
    group = dist.new_group(ranks=ranks)
    if rank in ranks:
      my_column = group
  return my_pair, my_pair_ranks, my_column


def run_uneven_pair_collectives() -> None:
  """Pair k issues k + 1 pair all-reduces; no other pair joins them."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(600)  # A hang here is the failure mode under test.
  my_pair, my_pair_ranks, _ = _make_groups(rank)
  pair_index = _PAIRS.index(my_pair_ranks)

  x = torch.tensor([1.0, float(rank)], device="tpu")
  for _ in range(pair_index + 1):
    dist.all_reduce(x, group=my_pair)
  out = x.cpu()
  # The first all-reduce sums the two members; every later one doubles.
  a, b = my_pair_ranks
  expected = torch.tensor([2.0, float(a + b)]) * (2 ** pair_index)
  logging.info("rank %d pair %s out=%s expected=%s", rank, my_pair_ranks, out,
               expected)
  utils.assert_close(actual=out, expected=expected)

  # Every rank meets again on the world group afterwards.
  y = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(y)
  utils.assert_close(y.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_interleaved_pair_and_world_collectives() -> None:
  """Each iteration: pair k runs k + 1 all-gathers, then a world all-reduce."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(900)
  my_pair, my_pair_ranks, my_column = _make_groups(rank)
  pair_index = _PAIRS.index(my_pair_ranks)
  a, b = my_pair_ranks

  for it in range(20):
    x = torch.full((4, 8), float(rank + it), device="tpu")
    for _ in range(pair_index + 1):
      gathered = torch.empty((8, 8), device="tpu")
      dist.all_gather_into_tensor(gathered, x, group=my_pair)
      x = gathered[:4] + gathered[4:]
    # After n rounds, x = (a + b + 2 * it) * 2 ** (n - 1).
    expected = torch.full(
        (4, 8), float(a + b + 2 * it) * (2 ** pair_index))
    utils.assert_close(x.cpu(), expected)

    # Column groups run different counts too (column 0 twice, column 1 once).
    # The first all-reduce sums the members; a second one multiplies the sum by
    # the group size.
    z = torch.tensor([float(rank)], device="tpu")
    column = _COLUMNS[rank % 2]
    column_rounds = 2 if rank % 2 == 0 else 1
    for _ in range(column_rounds):
      dist.all_reduce(z, group=my_column)
    expected_z = float(sum(column)) * len(column) ** (column_rounds - 1)
    utils.assert_close(z.cpu(), torch.tensor([expected_z]))

    w = torch.tensor([float(it)], device="tpu")
    dist.all_reduce(w)
    utils.assert_close(w.cpu(), torch.tensor([float(it * _WORLD)]))


def _stage_kernel(tag):
  """A shard_map kernel over a (stage, pair) mesh of the whole slice, tagged
  with one pair's world ranks: the pair sum of the input, position
  0's input handed to position 1, and a Pallas body's own pair position
  (axis_index inside a kernel body lowers to Mosaic's device id, the way a
  ring kernel finds its remote-DMA peer; at the top level it would be an
  HLO partition-id, which the SPMD partitioner rejects)."""
  import inspect

  import jax
  from jax.experimental import pallas as pl
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental.shard_map import shard_map
  from jax.sharding import Mesh, PartitionSpec
  from torch_tpu._internal.pallas import pallas as pallas_impl

  mesh = Mesh(np.asarray(jax.devices()[:_WORLD]).reshape(_WORLD // 2, 2),
              ("stage", "pair"))

  def position_body(x_ref, o_ref):
    column = jax.lax.axis_index("pair").astype(jnp.float32)
    o_ref[...] = jnp.zeros_like(x_ref[...]) + column

  def kernel(x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:

    def inner(xb):
      total = jax.lax.psum(xb, "pair")
      forwarded = jax.lax.ppermute(xb, "pair", [(0, 1)])
      column = pl.pallas_call(
          position_body,
          out_shape=jax.ShapeDtypeStruct(xb.shape, jnp.float32))(xb)
      return total, forwarded, column

    spec = PartitionSpec()
    return shard_map(inner, mesh=mesh, in_specs=(spec,),
                     out_specs=(spec, spec, spec), check_rep=False)(x)

  static_argnums = pallas_impl._infer_static_argnums(inspect.signature(kernel))
  callable_ = pallas_impl.JaxCallable(
      name="test::stage_kernel",
      jit_fn=jax.jit(kernel, static_argnums=static_argnums),
      trace_key=f"stage_kernel_{tag}",
      static_argnums=static_argnums,
      collective_devices=list(tag),
  )
  op = torch.library.custom_op("test::stage_kernel", callable_, mutates_args=())
  op.register_fake(lambda x: (torch.empty_like(x), torch.empty_like(x),
                              torch.empty_like(x)))
  return op


def run_uneven_tagged_kernels() -> None:
  """Pair k launches a kernel tagged with its devices k + 1 times."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(900)
  my_pair, my_pair_ranks, _ = _make_groups(rank)
  pair_index = _PAIRS.index(my_pair_ranks)
  a, b = my_pair_ranks

  # A device's logical id is its world rank.
  tag = list(my_pair_ranks)
  op = _stage_kernel(tag)

  position = my_pair_ranks.index(rank)
  for i in range(pair_index + 1):
    x = torch.full((128, 128), float(rank + 1 + 10 * i), device="tpu")
    total, forwarded, column = op(x)
    utils.assert_close(total.cpu(),
                       torch.full((128, 128), float(a + b + 2 + 20 * i)))
    # Position 1 holds position 0's block; position 0 holds zeros.
    utils.assert_close(
        forwarded.cpu(),
        torch.full((128, 128), float(a + 1 + 10 * i) if position else 0.0))
    utils.assert_close(column.cpu(), torch.full((128, 128), float(position)))

  # A c10d pair collective and the tagged kernel share one program.
  y = torch.full((128, 128), float(rank + 1), device="tpu")
  dist.all_reduce(y, group=my_pair)
  total, _, _ = op(y)
  utils.assert_close(total.cpu(), torch.full((128, 128), float(2 * (a + b + 2))))

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


class SubgroupLaunchOrderTest(seed_test_utils.MultiProcessRepeatableTest):

  def test_uneven_pair_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=singlehost_wrapper.tpu_env_wrapper(run_uneven_pair_collectives),
    )

  def test_interleaved_pair_and_world_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=singlehost_wrapper.tpu_env_wrapper(
            run_interleaved_pair_and_world_collectives),
    )

  def test_uneven_tagged_kernels(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=singlehost_wrapper.tpu_env_wrapper(run_uneven_tagged_kernels),
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
