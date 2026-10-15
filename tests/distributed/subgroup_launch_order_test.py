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

"""Tests that process subgroups launch their programs on their own.

Each subgroup's collectives and kernels are compiled for that subgroup's devices
only, so the members of one subgroup may issue any number of programs without
the other subgroups launching anything. Pipeline stages with tensor parallelism
inside each stage rely on this: every stage runs its own sequence of stage-local
collectives while all stages join only the collectives over the whole slice.
"""

import inspect
import os
import pathlib
import pickle
import shutil
import signal
from unittest import mock
import uuid

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
import torch
from torch import distributed as dist
from torch._dynamo.utils import counters
import torch._functorch.config as functorch_config
from torch._higher_order_ops import scan
from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir
from torch.distributed import _functional_collectives as funcol
import torch.multiprocessing as mp
from torch_tpu._internal import pallas
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import oss_utils
from tests import seed_test_utils
from tests.distributed import distributed_utils

P = jax.sharding.PartitionSpec

_WORLD = 8
_PAIRS = [[0, 1], [2, 3], [4, 5], [6, 7]]
_COLUMNS = [[0, 2, 4, 6], [1, 3, 5, 7]]
_HALVES = [[0, 1, 2, 3], [4, 5, 6, 7]]
# On a host of four two-core chips, a pair holds the two cores of one chip, a
# half both cores of two chips and a column one core of each chip, so halves
# and columns communicate across chips.
# The second group lists its ranks in descending order: its group ranks, and the
# order of its devices, run opposite to the first group's.
_REVERSED = [[0, 1, 2, 3], [7, 6, 5, 4]]
# With the host's rank-to-device order [0, 1, 4, 5, 6, 7, 2, 3], a diagonal
# holds both cores of two chips that differ along two axes, so its devices fill
# no box of chips; the second diagonal has no device on the origin chip.
_DIAGONALS = [[0, 1, 4, 5], [2, 3, 6, 7]]
_LAYOUTS = {
    "pairs": _PAIRS,
    "halves": _HALVES,
    "columns": _COLUMNS,
    "reversed": _REVERSED,
    "diagonals": _DIAGONALS,
}
_LAYOUT_CASES = tuple((name, name) for name in _LAYOUTS)
_TIER2_CACHE_ROOT = "/dev/shm/torch_tpu_cache"
# Groups in other than ascending rank order need `dist.new_group(sort_ranks=)`.
_HAS_SORT_RANKS = "sort_ranks" in inspect.signature(dist.new_group).parameters
_AXIS = "group"
# One table row per CSR index slot: 16 CSR rows of 64 slots, 8 of them used.
_TABLE_ROWS = 1024
_TABLE_WIDTH = 16
_CSR_ROWS = 16
_CSR_ROW_NON_ZEROES = 8
_CSR_MAX_ROW_NON_ZEROES = 64


def _make_groups(rank):
  """Creates every pair and column group on every rank; returns this rank's.

  PyTorch's `dist.new_group` must be called by every rank for every group;
  torch_tpu only uses this rank's own groups.
  """
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


def _make_layout_group(rank, layout):
  """Creates every group of `layout` on every rank; returns this rank's.

  PyTorch's `dist.new_group` must be called by every rank for every group;
  torch_tpu only uses this rank's own group.

  Returns:
    The group of this rank, its ranks and its index in `layout`.
  """
  my_group, my_ranks = None, None
  for ranks in _LAYOUTS[layout]:
    if ranks == sorted(ranks):
      group = dist.new_group(ranks=ranks)
    else:
      group = dist.new_group(ranks=ranks, sort_ranks=False)
    if rank in ranks:
      my_group, my_ranks = group, ranks
  return my_group, my_ranks, _LAYOUTS[layout].index(my_ranks)


def run_uneven_group_collectives(layout: str) -> None:
  """Group k issues k + 1 group all-reduces; no other group joins them."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)  # A hang here is the failure mode under test.
  my_group, my_ranks, group_index = _make_layout_group(rank, layout)
  members = len(my_ranks)

  x = torch.tensor([1.0, float(rank)], device="tpu")
  for _ in range(group_index + 1):
    dist.all_reduce(x, group=my_group)
  out = x.cpu()
  # The first all-reduce sums the members; every later one multiplies the sum
  # by the group size.
  expected = torch.tensor([float(members), float(sum(my_ranks))]) * (
      members**group_index
  )
  logging.info(
      "rank %d group %s out=%s expected=%s", rank, my_ranks, out, expected
  )
  utils.assert_close(actual=out, expected=expected)

  # The gathered rows follow the group's rank order.
  gathered = torch.empty(members, device="tpu")
  dist.all_gather_into_tensor(
      gathered, torch.tensor([float(rank)], device="tpu"), group=my_group
  )
  utils.assert_close(gathered.cpu(), torch.tensor([float(r) for r in my_ranks]))

  # Every rank meets again on the world group afterwards.
  y = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(y)
  utils.assert_close(y.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_interleaved_subgroup_and_world_collectives() -> None:
  """Each iteration: pair k runs k + 1 all-gathers, then a world all-reduce."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
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
    expected = torch.full((4, 8), float(a + b + 2 * it) * (2**pair_index))
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


def _group_sum_op(my_group):
  """A kernel over the mesh of one group: the sum of the group's inputs."""
  mesh = pallas.get_pallas_mesh(axis_names=(_AXIS,), process_group=my_group)

  def fn(x: jax.Array) -> jax.Array:
    return jax.shard_map(
        lambda block: jax.lax.psum(block, _AXIS),
        mesh=mesh,
        in_specs=(P(_AXIS),),
        out_specs=P(_AXIS),
        check_vma=False,
    )(x)

  return pallas.jax_op(
      "subgroup_launch_order::group_sum",
      fn,
      mesh=mesh,
      input_partition_specs=[P(_AXIS)],
      output_partition_specs=[P(_AXIS)],
  )


def run_uneven_group_kernels(layout: str) -> None:
  """Group k launches a kernel over its own mesh k + 1 times."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_group, my_ranks, group_index = _make_layout_group(rank, layout)
  members = len(my_ranks)
  # The sum of rank + 1 over the members.
  member_sum = float(sum(my_ranks) + members)
  op = _group_sum_op(my_group)

  # A kernel may run over the group's devices in another order. Over the
  # reversed mesh, group rank i sits at mesh position members - 1 - i, so a
  # shift to the next mesh position delivers the block of group rank i + 1.
  mesh = pallas.get_pallas_mesh(axis_names=(_AXIS,), process_group=my_group)
  reversed_mesh = jax.sharding.Mesh(
      mesh.devices[::-1], axis_names=mesh.axis_names
  )
  shift = [(p, (p + 1) % members) for p in range(members)]

  def shift_by_one(x: jax.Array) -> jax.Array:
    return jax.shard_map(
        lambda block: jax.lax.ppermute(block, _AXIS, perm=shift),
        mesh=reversed_mesh,
        in_specs=(P(_AXIS),),
        out_specs=P(_AXIS),
        check_vma=False,
    )(x)

  reordered = pallas.jax_op(
      "subgroup_launch_order::reordered",
      shift_by_one,
      mesh=reversed_mesh,
      input_partition_specs=[P(_AXIS)],
      output_partition_specs=[P(_AXIS)],
  )
  x = torch.full((1, 8, 128), float(rank + 1), device="tpu")
  next_rank = my_ranks[(my_ranks.index(rank) + 1) % members]
  utils.assert_close(
      reordered(x).cpu(), torch.full((1, 8, 128), float(next_rank + 1))
  )

  # Kernels over conflicting meshes cannot share one program.
  @spmd_util.spmd_safe
  def conflicting(y: torch.Tensor) -> torch.Tensor:
    return op(reordered(y))

  try:
    torch.compile(conflicting, backend="tpu", fullgraph=True, dynamic=False)(
        x
    ).cpu()
  except (RuntimeError, torch._dynamo.exc.BackendCompilerFailed) as e:
    assert (
        "expected Pallas kernels and scans in the same program to have the"
        " same device assignment"
    ) in str(e), e
  else:
    raise AssertionError("kernels over conflicting meshes shared a program")

  for i in range(group_index + 1):
    x = torch.full((1, 8, 128), float(rank + 1 + 10 * i), device="tpu")
    total = op(x)
    utils.assert_close(
        total.cpu(),
        torch.full((1, 8, 128), member_sum + 10 * i * members),
    )

  # A group collective and the group's kernel share one program.
  @spmd_util.spmd_safe
  def region(y: torch.Tensor) -> torch.Tensor:
    gathered = torch.empty((members, 8, 128), device=y.device)
    dist.all_gather_into_tensor(gathered, y, group=my_group)
    return op(gathered.sum(dim=0, keepdim=True))

  debugs = []
  compiled = torch.compile(
      region,
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={"serializable": False, "debug_callback": debugs.append},
  )
  y = torch.full((1, 8, 128), float(rank + 1), device="tpu")
  utils.assert_close(
      compiled(y).cpu(), torch.full((1, 8, 128), members * member_sum)
  )
  modules = [m for debug in debugs for m in debug.stablehlo_forward_text]
  assert any(
      "stablehlo.all_gather" in m and "stablehlo.all_reduce" in m
      for m in modules
  ), modules

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_uneven_group_sparse_core_programs(layout: str) -> None:
  """Group k launches a SparseCore gather and a group all-reduce k + 1 times."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_group, my_ranks, group_index = _make_layout_group(rank, layout)
  members = len(my_ranks)

  # CSR row j gathers the table rows 8 * j, ..., 8 * j + 7.
  non_zeroes = _CSR_ROWS * _CSR_ROW_NON_ZEROES
  row_pointers = (
      torch.arange(1, _CSR_ROWS + 1, dtype=torch.int32) * _CSR_ROW_NON_ZEROES
  ).to("tpu")
  indices = torch.full(
      (_CSR_ROWS * _CSR_MAX_ROW_NON_ZEROES,), 2**31 - 1, dtype=torch.int32
  )
  indices[:non_zeroes] = torch.arange(non_zeroes, dtype=torch.int32)
  indices = indices.to("tpu")
  table_rows = torch.arange(_TABLE_ROWS, dtype=torch.float32).unsqueeze(1)

  # The SparseCore op and the group collective share one program.
  # `serializable=False` compiles the program in every run instead of loading
  # it from the AOTAutograd cache.
  @torch.compile(backend="tpu", fullgraph=True, options={"serializable": False})
  @spmd_util.spmd_safe
  def gather_and_reduce(table: torch.Tensor) -> torch.Tensor:
    gathered = torch.tpu.ops.sparse_gather(
        row_pointers,
        indices,
        table,
        max_non_zeroes_per_row=_CSR_MAX_ROW_NON_ZEROES,
    )
    dist.all_reduce(gathered, group=my_group)
    return gathered

  for i in range(group_index + 1):
    # Row r of the table of a rank holds r + rank + 10 * i.
    table = (table_rows + float(rank + 10 * i)).repeat(1, _TABLE_WIDTH)
    out = gather_and_reduce(table.to("tpu")).cpu()[:non_zeroes]
    expected = (
        members * table_rows[:non_zeroes]
        + float(sum(my_ranks) + 10 * i * members)
    ).repeat(1, _TABLE_WIDTH)
    utils.assert_close(out, expected)

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def _pair_swap_op(my_pair):
  """A kernel over the mesh of one pair: each member gets the other's block."""
  mesh = pallas.get_pallas_mesh(axis_names=(_AXIS,), process_group=my_pair)

  def fn(x: jax.Array) -> jax.Array:
    return jax.shard_map(
        lambda block: jax.lax.ppermute(block, _AXIS, perm=[(0, 1), (1, 0)]),
        mesh=mesh,
        in_specs=(P(_AXIS),),
        out_specs=P(_AXIS),
        check_vma=False,
    )(x)

  return pallas.jax_op(
      "subgroup_launch_order::pair_swap",
      fn,
      mesh=mesh,
      input_partition_specs=[P(_AXIS)],
      output_partition_specs=[P(_AXIS)],
  )


def run_uneven_pair_scans() -> None:
  """Pair k runs a scan of pair collectives and pair kernels k + 1 times."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_pair, my_pair_ranks, _ = _make_groups(rank)
  pair_index = _PAIRS.index(my_pair_ranks)
  a, b = my_pair_ranks
  partner = b if rank == a else a
  swap = _pair_swap_op(my_pair)
  steps = 3

  def gather_step(carry: torch.Tensor, x: torch.Tensor):
    # Row 0 holds the block of group rank 0, row 1 that of group rank 1.
    gathered = funcol.all_gather_tensor(carry + x, 0, my_pair)
    total = gathered[0:1] + 2 * gathered[1:2]
    return total, total.clone()

  def swap_step(carry: torch.Tensor, x: torch.Tensor):
    total = carry + swap(x)
    return total, total.clone()

  gather_scan = torch.compile(scan, backend="tpu", dynamic=False)
  swap_scan = torch.compile(scan, backend="tpu", dynamic=False)
  for i in range(pair_index + 1):
    xs = torch.stack([
        torch.full((1, 8, 128), float(rank + 10 * i + s), device="tpu")
        for s in range(steps)
    ])
    # carry_s = 3 * carry_(s-1) + a's step-s input + 2 * b's step-s input.
    carry, _ = gather_scan(
        gather_step, torch.zeros((1, 8, 128), device="tpu"), xs, dim=0
    )
    expected = 0.0
    for s in range(steps):
      expected = 3 * expected + float(a + 2 * b + 3 * (10 * i + s))
    utils.assert_close(carry.cpu(), torch.full((1, 8, 128), expected))
    # Each member sums the partner's inputs.
    carry, _ = swap_scan(
        swap_step, torch.zeros((1, 8, 128), device="tpu"), xs, dim=0
    )
    expected = float(sum(partner + 10 * i + s for s in range(steps)))
    utils.assert_close(carry.cpu(), torch.full((1, 8, 128), expected))

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_mixed_programs_run_on_the_world() -> None:
  """Programs that combine a pair's collectives with world ops run on the world.

  The pair's collective is tiled over every pair, so all ranks take part.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_pair, _, _ = _make_groups(rank)

  @torch.compile(backend="tpu", fullgraph=True)
  @spmd_util.spmd_safe
  def mixed(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(a)
    dist.all_reduce(b, group=my_pair)
    return a + b

  x = torch.ones(4, device="tpu")
  y = torch.ones(4, device="tpu")
  utils.assert_close(
      mixed(x, y).cpu(), torch.full((4,), float(_WORLD + 2), dtype=torch.float32)
  )

  # A sparse-dense-matmul lookup shards its table across the SparseCores of
  # all devices (carrying the full-slice device assignment at dispatch time):
  # 16 samples of 2 ids each, in a table of 32 rows.
  csr = torch.ops.tpu.preprocess_sparse_dense_matmul_input(
      torch.arange(32, dtype=torch.int32),
      torch.arange(0, 33, 2, dtype=torch.int32),
      128,
      128,
      128,
      16,
      "sum",
      1,
      1,
      2,
      16,  # The SparseCore SIMD width of TPU v7.
      8,  # The HBM word size of TPU v7, in units of 4 bytes.
      True,
      "table_0",
  )
  csr = [
      csr[key].to("tpu")
      for key in ("row_pointers", "embedding_ids", "sample_ids", "gains")
  ]

  @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
  @spmd_util.spmd_safe
  def lookup(row_pointers, embedding_ids, sample_ids, gains, table):
    activations = torch.tpu.ops.sparse_dense_matmul(
        row_pointers,
        embedding_ids,
        sample_ids,
        gains,
        table,
        device_batch_size=16,
        max_ids_per_partition=32,
        max_unique_ids_per_partition=32,
    )
    dist.all_reduce(activations, group=my_pair)
    return activations

  activations = lookup(*csr, torch.zeros((32, 8), device="tpu")).cpu()
  utils.assert_close(activations, torch.zeros_like(activations))


def run_uneven_group_all_to_all(layout: str) -> None:
  """Group k exchanges uneven, partly empty splits k + 1 times.

  Every element is tagged with its sender, its position and the iteration, so
  an element at the wrong offset or left from an earlier iteration fails the
  check. Each iteration exchanges through `dist.all_to_all_single` eagerly and
  through `torch.tpu.ops.ragged_all_to_all` under `torch.compile`.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_group, my_ranks, group_index = _make_layout_group(rank, layout)
  members = len(my_ranks)
  member = my_ranks.index(rank)
  # Member i sends (i + j) % n elements to member j, so some splits are empty.
  sends = [[(i + j) % members for j in range(members)] for i in range(members)]
  send_sizes = sends[member]
  recv_sizes = [sends[j][member] for j in range(members)]
  input_offsets = [sum(send_sizes[:j]) for j in range(members)]
  # Where this member's elements land in each receiver's output.
  output_offsets = [
      sum(sends[i][j] for i in range(member)) for j in range(members)
  ]

  def tagged(sender: int, start: int, count: int, it: int) -> torch.Tensor:
    return (
        my_ranks[sender] * 10000
        + it * 100
        + torch.arange(start, start + count, dtype=torch.int32)
    )

  @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
  def ragged(x, out, in_offsets, sizes_out, out_offsets, sizes_in):
    return torch.tpu.ops.ragged_all_to_all(
        x,
        out,
        in_offsets,
        sizes_out,
        out_offsets,
        sizes_in,
        my_group.group_name,
    )

  def as_tpu(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32).tpu()

  for it in range(group_index + 1):
    x = tagged(member, 0, sum(send_sizes), it).tpu()
    expected = torch.cat([
        tagged(j, sum(sends[j][:member]), sends[j][member], it)
        for j in range(members)
    ])
    out = torch.zeros(sum(recv_sizes), dtype=torch.int32, device="tpu")
    dist.all_to_all_single(out, x, recv_sizes, send_sizes, group=my_group)
    utils.assert_close(out.cpu(), expected)
    out = ragged(
        x,
        torch.zeros(sum(recv_sizes), dtype=torch.int32, device="tpu"),
        as_tpu(input_offsets),
        as_tpu(send_sizes),
        as_tpu(output_offsets),
        as_tpu(recv_sizes),
    )
    utils.assert_close(out.cpu(), expected)

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def _half_region(my_half, members):
  """An all-gather over one half, then a reduce-scatter of the gathered rows."""

  @spmd_util.spmd_safe
  def region(x: torch.Tensor) -> torch.Tensor:
    gathered = torch.empty((members * 4, 8), device=x.device)
    dist.all_gather_into_tensor(gathered, x, group=my_half)
    scattered = torch.empty((4, 8), device=x.device)
    dist.reduce_scatter_tensor(scattered, gathered, group=my_half)
    return scattered

  return region


def run_uneven_compiled_half_programs() -> None:
  """Half k launches a compiled all-gather and reduce-scatter k + 1 times."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_half, my_half_ranks, half_index = _make_layout_group(rank, "halves")
  members = len(my_half_ranks)
  region = _half_region(my_half, members)

  # `serializable=False` keeps this a fresh compile, whose executables the
  # debug callback reports, rather than an AOTAutograd cache hit.
  debugs = []
  compiled = torch.compile(
      region,
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={"serializable": False, "debug_callback": debugs.append},
  )
  for i in range(half_index + 1):
    value = float(rank + 1 + 10 * i)
    x = torch.full((4, 8), value, device="tpu")
    # Every member gathers the same rows, so the sum returns its own rows once
    # per member.
    utils.assert_close(compiled(x).cpu(), torch.full((4, 8), members * value))

  # A pickled executable loads back and runs on the devices of its half, also
  # for the half off the origin chip.
  (artifact,) = debugs[0].compiled_executables
  restored = pickle.loads(pickle.dumps(artifact))
  x = torch.full((4, 8), float(rank + 1), device="tpu")
  (out,) = restored(x)
  utils.assert_close(out.cpu(), torch.full((4, 8), members * float(rank + 1)))

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_cached_half_programs(warm: bool) -> None:
  """Compiles the half programs with the AOTAutograd cache on."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_half, my_half_ranks, half_index = _make_layout_group(rank, "halves")
  members = len(my_half_ranks)
  region = _half_region(my_half, members)
  compiled = torch.compile(region, backend="tpu", fullgraph=True, dynamic=False)
  for i in range(half_index + 1):
    value = float(rank + 1 + 10 * i)
    x = torch.full((4, 8), value, device="tpu")
    utils.assert_close(compiled(x).cpu(), torch.full((4, 8), members * value))

  aot = counters["aot_autograd"]
  if warm:
    # Both halves reload their programs, the half off the origin at its chips.
    assert aot["autograd_cache_hit"] == 1, dict(aot)
  else:
    assert aot["autograd_cache_bypass"] == 0, dict(aot)

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_strict_cache_half_programs() -> None:
  """A strict AOTAutograd cache saves the programs of both halves."""
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_half, my_half_ranks, _ = _make_layout_group(rank, "halves")
  members = len(my_half_ranks)
  region = _half_region(my_half, members)
  x = torch.full((4, 8), float(rank + 1), device="tpu")
  with functorch_config.patch(strict_autograd_cache=True):
    compiled = torch.compile(
        region, backend="tpu", fullgraph=True, dynamic=False
    )
    utils.assert_close(
        compiled(x).cpu(), torch.full((4, 8), members * float(rank + 1))
    )

  w = torch.tensor([float(rank)], device="tpu")
  dist.all_reduce(w)
  utils.assert_close(w.cpu(), torch.tensor([float(sum(range(_WORLD)))]))


def run_group_programs_from_the_caches(layout: str, compilations: int) -> None:
  """The members of each group compile the group's program `compilations` times.

  The members that do not compile it load it from the tier-2 or the tier-3
  cache, a group that fills a box of devices off the origin at its own chips;
  a member whose load fails compiles the program again.
  """
  dist.init_process_group(backend="tpu_dist")
  rank = int(os.environ["RANK"])
  assert int(os.environ["WORLD_SIZE"]) == _WORLD
  signal.alarm(45)
  my_group, my_ranks, _ = _make_layout_group(rank, layout)

  x = torch.full((3, 5), float(rank + 1), device="tpu")
  x.cpu()
  stats = torch.tpu._get_cache_stats  # pylint: disable=protected-access
  before = stats().num_compilations
  dist.all_reduce(x, group=my_group)
  out = x.cpu()
  local_compilations = stats().num_compilations - before
  utils.assert_close(
      out, torch.full((3, 5), float(sum(r + 1 for r in my_ranks)))
  )

  counts = torch.empty(_WORLD, dtype=torch.int32, device="tpu")
  dist.all_gather_into_tensor(
      counts,
      torch.tensor([local_compilations], dtype=torch.int32, device="tpu"),
  )
  counts = counts.cpu().tolist()
  logging.info("rank %d compilations per rank %s", rank, counts)
  for group in _LAYOUTS[layout]:
    assert sum(counts[r] for r in group) == compilations, counts


@oss_utils.skip_if_libtpu_version(
    lambda v: v < (0, 0, 47),
    "process subgroups launching on their own need libtpu 0.0.47 or newer",
)
class SubgroupLaunchOrderTest(seed_test_utils.MultiProcessRepeatableTest):

  def _skip_unsupported_layout(self, layout: str) -> None:
    if layout == "reversed" and not _HAS_SORT_RANKS:
      self.skipTest("dist.new_group has no sort_ranks in this torch version")

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_uneven_group_collectives(self, layout: str):
    self._skip_unsupported_layout(layout)
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_uneven_group_collectives,
        layout=layout,
    )

  def test_interleaved_subgroup_and_world_collectives(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_interleaved_subgroup_and_world_collectives,
    )

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_uneven_group_kernels(self, layout: str):
    self._skip_unsupported_layout(layout)
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_uneven_group_kernels,
        layout=layout,
    )

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_uneven_group_sparse_core_programs(self, layout: str):
    self._skip_unsupported_layout(layout)
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_uneven_group_sparse_core_programs,
        layout=layout,
    )

  @parameterized.named_parameters(*_LAYOUT_CASES)
  def test_uneven_group_all_to_all(self, layout: str):
    self._skip_unsupported_layout(layout)
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_uneven_group_all_to_all,
        layout=layout,
    )

  def test_uneven_pair_scans(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_uneven_pair_scans,
    )

  def test_mixed_programs_run_on_the_world(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_mixed_programs_run_on_the_world,
    )

  def test_compiled_half_programs_through_the_aot_cache(self):
    with temporary_cache_dir(self.create_tempdir().full_path):
      for warm in (False, True):
        distributed_utils.dist_run(
            nproc_per_node=_WORLD,
            fn=run_cached_half_programs,
            warm=warm,
        )

  def test_strict_cache_saves_half_programs(self):
    with temporary_cache_dir(self.create_tempdir().full_path):
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_strict_cache_half_programs,
      )

  def _fresh_tier2_cache(self) -> str:
    """Returns the name of a tier-2 cache that no other test uses."""
    name = f"subgroup_launch_order_test_{uuid.uuid4().hex}"
    self.addCleanup(
        shutil.rmtree, os.path.join(_TIER2_CACHE_ROOT, name), ignore_errors=True
    )
    return name

  @parameterized.named_parameters(
      ("halves", "halves"), ("diagonals", "diagonals")
  )
  def test_group_programs_through_the_tier2_cache(self, layout: str):
    # No tier-3 cache, whose backup compilations would count.
    tier2_cache = self._fresh_tier2_cache()
    with mock.patch.dict(
        os.environ, {"TORCH_TPU_TIER2_COMPILATION_CACHE": tier2_cache}
    ):
      os.environ.pop("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", None)
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_group_programs_from_the_caches,
          layout=layout,
          compilations=1,
      )

  @parameterized.named_parameters(
      ("halves", "halves"), ("diagonals", "diagonals")
  )
  def test_group_programs_through_the_tier3_cache(self, layout: str):
    tier2_cache = self._fresh_tier2_cache()
    with mock.patch.dict(
        os.environ,
        {
            "TORCH_TPU_TIER2_COMPILATION_CACHE": tier2_cache,
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": (
                self.create_tempdir().full_path
            ),
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
        },
    ):
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_group_programs_from_the_caches,
          layout=layout,
          compilations=1,
      )
      # Without the tier-2 entries, the second run loads from the tier-3 cache.
      for path in pathlib.Path(_TIER2_CACHE_ROOT, tier2_cache).glob("**/*.bin"):
        path.unlink()
      distributed_utils.dist_run(
          nproc_per_node=_WORLD,
          fn=run_group_programs_from_the_caches,
          layout=layout,
          compilations=0,
      )

  def test_uneven_compiled_half_programs(self):
    distributed_utils.dist_run(
        nproc_per_node=_WORLD,
        fn=run_uneven_compiled_half_programs,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")
  multiprocessing.handle_test_main(absltest.main)
