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

"""Tests SparseCore programs on all devices of a host, alone and with collectives.

A sparse-dense-matmul lookup carries the device assignment of the whole slice,
since its table is sharded across the SparseCores of all devices; a per-device
SparseCore op such as `sparse_gather` carries none. Either runs in one program
with a collective over the world, whose devices follow the world rank order.
"""

from collections.abc import Sequence
from absl.testing import absltest
import torch
from torch import distributed as dist
from torch_tpu._internal.distributed import spmd_util
from torch_tpu._internal.utils import test_utils as utils
from torch_tpu._internal.distributed import multiprocessing
from tests import oss_utils
from tests.distributed import continuous_test_utils

_WORLD = 8

# The sparse-dense-matmul lookup. TPU v7 has two SparseCores per device and a
# SparseCore SIMD width of 16.
_NUM_SC_PER_DEVICE = 2
_SC_SIMD_WIDTH = 16
# The HBM word size of TPU v7, in units of 4 bytes.
_HBM_WORD_SIZE_IN_4B = 8
_NUM_SHARDS = _WORLD * _NUM_SC_PER_DEVICE
_VOCAB_SIZE = 64
_EMBEDDING_DIM = 8
# Samples per device, one id each.
_BATCH_SIZE = 8
_MAX_IDS_PER_PARTITION = 16
_MAX_UNIQUE_IDS_PER_PARTITION = 16
# The largest COO buffer one device can need: _MAX_IDS_PER_PARTITION ids for
# each of the _NUM_SHARDS partitions, from each of its SparseCores.
_COO_BUFFER_SIZE = _MAX_IDS_PER_PARTITION * _NUM_SC_PER_DEVICE * _NUM_SHARDS

# The sparse gather: 16 CSR rows of 64 slots, 8 of them used, and one table row
# per CSR index slot.
_TABLE_ROWS = 1024
_TABLE_WIDTH = 16
_CSR_ROWS = 16
_CSR_ROW_NON_ZEROES = 8
_CSR_MAX_ROW_NON_ZEROES = 64


def _lookup_ids(rank: int) -> torch.Tensor:
  """The id of each sample of `rank`.

  Since 9 is odd, the 8 samples of a rank hit 8 distinct shards (id % 16), one
  on each device, so every device serves every rank.

  Args:
    rank: The world rank whose samples to return.

  Returns:
    A 1D int32 tensor of _BATCH_SIZE ids.
  """
  return torch.tensor(
      [(8 * rank + 9 * s) % _VOCAB_SIZE for s in range(_BATCH_SIZE)],
      dtype=torch.int32,
  )


def _global_table() -> torch.Tensor:
  """Row g holds g + 1, so every row differs and none is all zeros."""
  return (
      torch.arange(1, _VOCAB_SIZE + 1, dtype=torch.float32)
      .unsqueeze(1)
      .repeat(1, _EMBEDDING_DIM)
  )


def _local_table(rank: int) -> torch.Tensor:
  """The rows of the global table that `rank` holds.

  Mod sharding, as in torch_tpu_embedding's distributed training test: shard i
  holds the ids g with g % _NUM_SHARDS == i, and rank r holds the shards 2 * r
  and 2 * r + 1.

  Args:
    rank: The world rank whose rows to return.

  Returns:
    A (_VOCAB_SIZE // _WORLD, _EMBEDDING_DIM) float32 tensor.
  """
  table = _global_table()
  sharded = torch.cat([table[i::_NUM_SHARDS] for i in range(_NUM_SHARDS)])
  rows_per_rank = _VOCAB_SIZE // _WORLD
  return sharded[rank * rows_per_rank : (rank + 1) * rows_per_rank]


def _expected_activations(rank: int) -> torch.Tensor:
  """The activations of `rank`: the global table row of each sample's id."""
  return _global_table()[_lookup_ids(rank).long()]


def _lookup_inputs(rank: int) -> Sequence[torch.Tensor]:
  """The CSR inputs and the local table shard of `rank`, on the TPU."""
  csr = torch.ops.tpu.preprocess_sparse_dense_matmul_input(
      _lookup_ids(rank),
      torch.arange(_BATCH_SIZE + 1, dtype=torch.int32),
      _MAX_IDS_PER_PARTITION,
      _MAX_UNIQUE_IDS_PER_PARTITION,
      _COO_BUFFER_SIZE,
      _BATCH_SIZE,
      "sum",
      1,  # local_device_count
      _WORLD,  # global_device_count
      _NUM_SC_PER_DEVICE,
      _SC_SIMD_WIDTH,
      _HBM_WORD_SIZE_IN_4B,
      False,  # allow_id_dropping: fail rather than drop an id.
      "table_0",
  )
  inputs = [
      csr[key]
      for key in ("row_pointers", "embedding_ids", "sample_ids", "gains")
  ]
  inputs.append(_local_table(rank))
  return [t.to("tpu") for t in inputs]


def _lookup(
    row_pointers: torch.Tensor,
    embedding_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    gains: torch.Tensor,
    table: torch.Tensor,
) -> torch.Tensor:
  return torch.tpu.ops.sparse_dense_matmul(
      row_pointers,
      embedding_ids,
      sample_ids,
      gains,
      table,
      device_batch_size=_BATCH_SIZE,
      max_ids_per_partition=_MAX_IDS_PER_PARTITION,
      max_unique_ids_per_partition=_MAX_UNIQUE_IDS_PER_PARTITION,
  )


def run_sparse_dense_matmul_lookup() -> None:
  """A compiled lookup into a table sharded across all devices."""
  rank = dist.get_rank()
  assert dist.get_world_size() == _WORLD

  compiled = torch.compile(
      _lookup, backend="tpu", fullgraph=True, dynamic=False
  )
  activations = compiled(*_lookup_inputs(rank))
  utils.assert_close(activations.cpu(), _expected_activations(rank))


def run_sparse_dense_matmul_with_world_all_reduce() -> None:
  """A lookup and a world all-reduce of its activations share one program."""
  rank = dist.get_rank()
  assert dist.get_world_size() == _WORLD

  # `serializable=False` compiles the program in every run instead of loading
  # it from the AOTAutograd cache.
  @torch.compile(
      backend="tpu",
      fullgraph=True,
      dynamic=False,
      options={"serializable": False},
  )
  @spmd_util.spmd_safe
  def lookup_and_reduce(
      row_pointers: torch.Tensor,
      embedding_ids: torch.Tensor,
      sample_ids: torch.Tensor,
      gains: torch.Tensor,
      table: torch.Tensor,
  ) -> torch.Tensor:
    activations = _lookup(row_pointers, embedding_ids, sample_ids, gains, table)
    dist.all_reduce(activations)
    return activations

  out = lookup_and_reduce(*_lookup_inputs(rank))
  expected = sum(_expected_activations(r) for r in range(_WORLD))
  utils.assert_close(out.cpu(), expected)


def run_sparse_gather_with_world_all_reduce() -> None:
  """A SparseCore gather and a world all-reduce share one program."""
  rank = dist.get_rank()
  assert dist.get_world_size() == _WORLD

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
    dist.all_reduce(gathered)
    return gathered

  # Row r of the table of a rank holds r + rank.
  table = (table_rows + float(rank)).repeat(1, _TABLE_WIDTH)
  out = gather_and_reduce(table.to("tpu")).cpu()[:non_zeroes]
  expected = (
      _WORLD * table_rows[:non_zeroes] + float(sum(range(_WORLD)))
  ).repeat(1, _TABLE_WIDTH)
  utils.assert_close(out, expected)


class SparseCoreCollectivesTest(
    continuous_test_utils.TpuMultiProcContinuousTest
):
  world_size = _WORLD

  def test_sparse_dense_matmul_lookup(self):
    run_sparse_dense_matmul_lookup()

  @oss_utils.skip_if_libtpu_version(
      lambda v: v < (0, 0, 50),
      "TODO(b/572969237): Requires libtpu >= 0.0.50 to repack SparseGather"
      " operands flattened by partitioning.",
  )
  def test_sparse_gather_with_world_all_reduce(self):
    run_sparse_gather_with_world_all_reduce()

  def test_sparse_dense_matmul_with_world_all_reduce(self):
    run_sparse_dense_matmul_with_world_all_reduce()


if __name__ == "__main__":
  multiprocessing.handle_test_main(absltest.main)
