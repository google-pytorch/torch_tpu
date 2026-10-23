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

"""Unit tests for process_group_utils."""

import operator

from absl.testing import absltest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed import fake_pg
from torch_tpu._internal.distributed import process_group_utils
from tests import seed_test_utils

FakeStore = fake_pg.FakeStore
ProcessGroupId = process_group_utils.ProcessGroupId


class ProcessGroupUtilsTest(seed_test_utils.RepeatableTest):

  def setUp(self) -> None:
    super().setUp()
    self.world_size = 4
    if not dist.is_initialized():
      dist.init_process_group(
          backend="fake",
          store=FakeStore(),
          rank=0,
          world_size=self.world_size,
      )

  def tearDown(self) -> None:
    if dist.is_initialized():
      dist.destroy_process_group()
    super().tearDown()

  def test_extract_process_group_id_non_collective_op_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(operator.add, args=(x, x))
    with self.assertRaisesRegex(
        ValueError,
        "expected node target to be in .* torch_tpu bug",
    ):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_empty_ranks_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", []),
    )
    with self.assertRaisesRegex(ValueError, "ranks cannot be empty"):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_missing_group_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum"),
    )
    with self.assertRaisesRegex(
        ValueError, "Could not infer process group argument"
    ):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_unsupported_group_type_raises(self) -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", 3.14),
    )
    with self.assertRaisesRegex(ValueError, "Unsupported process group type"):
      process_group_utils._extract_process_group_id_from_node(node)

  def test_extract_process_group_id_uninitialized_dist_raises(self) -> None:
    dist.destroy_process_group()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node_str = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", "0"),
    )
    node_int = graph.call_function(
        torch.ops._c10d_functional.all_reduce,
        args=(x, "sum", 0),
    )
    with self.assertRaisesRegex(
        RuntimeError, "Could not resolve the process group"
    ):
      process_group_utils._extract_process_group_id_from_node(node_str)
    with self.assertRaisesRegex(
        RuntimeError, "Could not resolve the process group"
    ):
      process_group_utils._extract_process_group_id_from_node(node_int)


class ProcessGroupIdTest(seed_test_utils.RepeatableTest):

  def test_empty_ranks_raises_value_error(self) -> None:
    with self.assertRaisesRegex(ValueError, "ranks cannot be empty"):
      ProcessGroupId([])

  def test_negative_rank_raises_value_error(self) -> None:
    with self.assertRaisesRegex(ValueError, "All ranks must be non-negative"):
      ProcessGroupId([-1, 0])

  def test_invalid_world_size_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "world_size must be strictly positive"
    ):
      ProcessGroupId([0, 1], world_size=0)
    with self.assertRaisesRegex(
        ValueError, "world_size must be strictly positive"
    ):
      ProcessGroupId([0, 1], world_size=-2)

  def test_rank_out_of_world_size_raises_value_error(self) -> None:
    with self.assertRaisesRegex(ValueError, r"All ranks must be in \[0, 4\)"):
      ProcessGroupId([0, 4], world_size=4)
    with self.assertRaisesRegex(ValueError, r"All ranks must be in \[0, 2\)"):
      ProcessGroupId([0, 2], world_size=2)

  def test_duplicate_ranks_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "ranks must not contain duplicates"
    ):
      ProcessGroupId([0, 1, 1])

  def test_unsorted_ranks_raises_value_error(self) -> None:
    with self.assertRaisesRegex(
        ValueError, "ranks must be sorted in ascending order"
    ):
      ProcessGroupId([1, 0])
    with self.assertRaisesRegex(
        ValueError, "ranks must be sorted in ascending order"
    ):
      ProcessGroupId([0, 3, 2])


if __name__ == "__main__":
  absltest.main()
