# Copyright 2025 Google LLC
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

import os
import re
from typing import TypeAlias

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal import testing as tt_testing

EagerMode: TypeAlias = execution_mode.EagerMode


class InternalSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.old_eager_mode = execution_mode.eager_mode
    execution_mode.eager_mode = EagerMode.DEFER_AND_FUSE

  def tearDown(self):
    execution_mode.eager_mode = self.old_eager_mode
    super().tearDown()

  def test_sync_no_wait_tensor(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(10, device=torch.device("tpu"))
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materializing(tensor))
      self.assertFalse(sync.is_materialized(tensor))

    sync.synchronize(y, wait=False)

    # Strict ordering forces x to be materialized along with y.
    # But it may not be ready yet.
    self.assertTrue(sync.is_materializing(x))

    # y was materialized, but may not be ready yet.
    self.assertTrue(sync.is_materializing(y))

    # z was not materialized because it was dispatched after the synchronized
    # tensor.
    self.assertFalse(sync.is_materializing(z))

  def test_sync_no_wait_list(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(10, device=torch.device("tpu"))
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materializing(tensor))
      self.assertFalse(sync.is_materialized(tensor))

    sync.synchronize([x, y, z], wait=False)

    # Everything is materialized, but may or may not be ready.
    for tensor in [x, y, z]:
      self.assertTrue(sync.is_materializing(tensor))

  def test_sync_and_wait_tensor(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(10, device=torch.device("tpu"))
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materializing(tensor))
      self.assertFalse(sync.is_materialized(tensor))

    sync.synchronize(y, wait=True)

    # Strict ordering forces x to be materialized along with y.
    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materialized(x))

    # y was materialized and is ready.
    self.assertTrue(sync.is_materializing(y))
    self.assertTrue(sync.is_materialized(y))

    # z was not materialized because it was dispatched after the synchronized
    # tensor.
    self.assertFalse(sync.is_materializing(z))

  def test_sync_and_wait_list(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(10, device=torch.device("tpu"))
    z = x + y

    # Nothing is materialized or ready.
    self.assertFalse(sync.is_materializing(x))
    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materializing(y))
    self.assertFalse(sync.is_materialized(y))
    self.assertFalse(sync.is_materializing(z))
    self.assertFalse(sync.is_materialized(z))

    sync.synchronize([x, y, z], wait=True)

    # Everything is materialized and ready.
    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materializing(y))
    self.assertTrue(sync.is_materialized(y))
    self.assertTrue(sync.is_materializing(z))
    self.assertTrue(sync.is_materialized(z))

  def test_sync_no_wait_all(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(11, device=torch.device("tpu"))
    z = torch.ones(12, device=torch.device("tpu"))

    # Nothing is materialized or ready.
    self.assertFalse(sync.is_materializing(x))
    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materializing(y))
    self.assertFalse(sync.is_materialized(y))
    self.assertFalse(sync.is_materializing(z))
    self.assertFalse(sync.is_materialized(z))

    sync.synchronize(wait=False)

    # Everything is materialized, but may or may not be ready.
    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materializing(y))
    self.assertTrue(sync.is_materializing(z))

  def test_sync_and_wait_all_materialized(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(11, device=torch.device("tpu"))
    z = torch.ones(12, device=torch.device("tpu"))

    self.assertFalse(sync.is_materializing(x))
    self.assertFalse(sync.is_materializing(y))
    self.assertFalse(sync.is_materializing(z))

    sync.synchronize(wait=True)

    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materializing(y))
    self.assertTrue(sync.is_materializing(z))

  def test_sync_and_wait_all_ready(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(11, device=torch.device("tpu"))
    z = torch.ones(12, device=torch.device("tpu"))

    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))
    self.assertFalse(sync.is_materialized(z))

    sync.synchronize(wait=True)

    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materialized(y))
    self.assertTrue(sync.is_materialized(z))

  def test_host_to_device_is_materializing(self):
    x = torch.ones(128, device="cpu").to(torch.device("tpu"))

    self.assertTrue(sync.is_materializing(x))

    sync.synchronize(x, wait=True)

    self.assertTrue(sync.is_materialized(x))

  def test_sync_with_zero_sized_tensor_on_tpu(self):
    # Create a zero-sized tensor on the TPU.
    tensor = torch.ones(2, 0, 3, dtype=torch.int32, device=torch.device("tpu"))

    # It is in a deferred state (constant zero-sized).
    self.assertFalse(sync.is_materializing(tensor))
    self.assertFalse(sync.is_materialized(tensor))

    sync.synchronize(tensor, wait=True)

    # After synchronization, it should be materialized and ready.
    self.assertTrue(sync.is_materializing(tensor))
    self.assertTrue(sync.is_materialized(tensor))

  def test_sync_with_materialized_zero_sized_tensor(self):
    # Create a zero-sized tensor on the CPU.
    tensor_cpu = torch.ones(2, 0, 3, dtype=torch.int32, device="cpu")

    # Send it to the TPU. This should create a deferred zero-sized constant
    # instead of actually transferring 0 bytes.
    tensor = tensor_cpu.to(torch.device("tpu"))
    self.assertFalse(sync.is_materializing(tensor))
    self.assertFalse(sync.is_materialized(tensor))

    sync.synchronize(tensor, wait=True)

    # After synchronization, it should be materialized and ready.
    self.assertTrue(sync.is_materializing(tensor))
    self.assertTrue(sync.is_materialized(tensor))

  def test_sync_list_with_empty_and_non_empty(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y_cpu = torch.ones(10, 0, device="cpu")
    y = y_cpu.to(torch.device("tpu"))
    # Should not raise error.
    sync.synchronize([x, y], wait=True)
    self.assertTrue(sync.is_materialized(x))

  def test_is_materializing_not_on_tpu(self):
    x = torch.ones(10, device=torch.device("cpu"))
    with self.assertRaisesRegex(
        RuntimeError,
        "tensor is not on the PrivateUse1 device",
    ):
      sync.is_materializing(x)

  def test_is_materialized_not_on_tpu(self):
    x = torch.ones(10, device=torch.device("cpu"))
    with self.assertRaisesRegex(
        RuntimeError,
        "tensor is not on the PrivateUse1 device",
    ):
      sync.is_materialized(x)

  def extract_graphviz_invariants(self, graphviz_string):
    node_params = []
    num_lines = 0
    for line in graphviz_string.split("\n"):
      node_match = re.fullmatch(r"^\d+ (\[.*\])", line)
      if node_match:
        node_params.append(node_match.group(1))
        continue
      num_lines += 1
    return node_params, num_lines

  def test_computation_graphviz_simple(self):
    x = torch.rand(2, 3, device=torch.device("tpu"))
    y = x**2
    z = x.sum()
    expected_graphviz_string = """Graphviz string: (try pasting in http://graphviz/ to see the graph)
digraph {
  // Vertices:
  0 [shape="box", label=" float32[] (materialized)"];
  1 [label="fill_.Scalar"];
  2 [shape="box", label=" uint64[2]"];
  3 [label="set_current_seed"];
  4 [shape="box", label=" uint64[2]"];
  5 [label="set_offset"];
  6 [shape="box", label=" uint64[2]"];
  7 [label="uniform_"];
  8 [shape="box", label=" uint64[2]"];
  9 [shape="box", label="x: float32[2, 3]"];
  10 [label="sum.IntList_out"];
  11 [shape="box", label="z: float32[]"];
  12 [label="pow.out"];
  13 [shape="box", label="y: float32[2, 3]"];

  // Edges:
  1 -> 2
  2 -> 3
  3 -> 4
  4 -> 5
  5 -> 6
  6 -> 7
  7 -> 8
  7 -> 9
  9 -> 10
  10 -> 11
  9 -> 12
  0 -> 12
  12 -> 13
}
"""

    node_params, num_lines = self.extract_graphviz_invariants(
        expected_graphviz_string
    )
    s = sync.computation_graphviz(y, z)
    for node_param in node_params:
      self.assertRegex(s, node_param)
    self.assertLen(s.split("\n"), num_lines)

  # TODO(bawilson): remove leaf node materialization
  @absltest.skip("Safe rule with leaf nodes forces everything to materialize")
  def test_computation_graphviz_partially_materialized(self):
    expected_before = """Graphviz string: (try pasting in http://graphviz/ to see the graph)
digraph {
  // Vertices:
  0 [shape="box", label="z_ones: float32[2, 4] (materialized)"];
  1 [shape="box", label="four: int64[] (materialized)"];
  2 [shape="box", label="y_ones: float32[3, 4] (materialized)"];
  3 [shape="box", label="three: int64[] (materialized)"];
  4 [shape="box", label="x_ones: float32[2, 3] (materialized)"];
  5 [shape="box", label="two: int64[] (materialized)"];
  6 [label="mul"];
  7 [shape="box", label="z: float32[2, 4]"];
  8 [label="mul"];
  9 [shape="box", label="y: float32[3, 4]"];
  10 [label="mul"];
  11 [shape="box", label="x: float32[2, 3]"];
  12 [label="mm.out"];
  13 [shape="box", label="x_times_y: float32[2, 4]"];
  14 [label="add.out"];
  15 [shape="box", label="w: float32[2, 4]"];

  // Edges:
  0 -> 6
  1 -> 6
  6 -> 7
  2 -> 8
  3 -> 8
  8 -> 9
  4 -> 10
  5 -> 10
  10 -> 11
  11 -> 12
  9 -> 12
  12 -> 13
  13 -> 14
  7 -> 14
  14 -> 15
}
"""

    expected_after = """Graphviz string: (try pasting in http://graphviz/ to see the graph)
digraph {
  // Vertices:
  0 [shape="box", label="w: float32[2, 4] (materialized)"];
  1 [shape="box", label="y_ones: float32[3, 4] (materialized)"];
  2 [shape="box", label="three: int64[] (materialized)"];
  3 [label="mul"];
  4 [shape="box", label="y: float32[3, 4]"];

  // Edges:
  1 -> 3
  2 -> 3
  3 -> 4
}
"""
    # Create materialized leaf inputs by doing a host to device copy.
    x_ones = torch.ones(2, 3, device="cpu").to(torch.device("tpu"))
    y_ones = torch.ones(3, 4, device="cpu").to(torch.device("tpu"))
    z_ones = torch.ones(2, 4, device="cpu").to(torch.device("tpu"))

    # Create a graph of deferred operations.
    x = x_ones * 2
    y = y_ones * 3
    z = z_ones * 4
    w = x @ y + z

    # Drop everything but the final outputs.
    del x_ones, y_ones, z_ones, z

    node_params, num_lines = self.extract_graphviz_invariants(expected_before)
    s = sync.computation_graphviz(y, w)
    for node_param in node_params:
      self.assertRegex(s, node_param)
    self.assertLen(s.split("\n"), num_lines)

    print(x.cpu())

    node_params, num_lines = self.extract_graphviz_invariants(expected_after)
    s = sync.computation_graphviz(y, w)

    for node_param in node_params:
      self.assertRegex(s, node_param)
    self.assertLen(s.split("\n"), num_lines)

  def test_dump_computation_graphviz(self):
    x = torch.ones(2, 3, device=torch.device("tpu"))
    y = torch.ones(2, 3, device=torch.device("tpu"))
    z = x + y
    expected_graphviz_string = """Graphviz string: (try pasting in http://graphviz/ to see the graph)
digraph {
  // Vertices:
  0 [label="fill_.Scalar"];
  1 [shape="box", label=" float32[2, 3]"];
  2 [label="fill_.Scalar"];
  3 [shape="box", label=" float32[2, 3]"];
  4 [label="add.out"];
  5 [shape="box", label=" float32[2, 3]"];

  // Edges:
  0 -> 1
  2 -> 3
  3 -> 4
  1 -> 4
  4 -> 5
}
"""
    output_dir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "")
    self.assertNotEmpty(output_dir)
    file_path = os.path.join(output_dir, "graphviz_dump.txt")
    sync.dump_computation_graphviz([y, z], file_path)

    with open(file_path, "r") as f:
      s = f.read()

    node_params, num_lines = self.extract_graphviz_invariants(
        expected_graphviz_string
    )
    for node_param in node_params:
      self.assertRegex(s, node_param)
    self.assertLen(s.split("\n"), num_lines)

  def test_computation_mlir_pytree(self):
    x = torch.ones(2, 3, device=torch.device("tpu"))
    y = x + 1.0

    # Passing a complex pytree (dict, list, tuple) to computation_mlir
    pytree_input = {"x": x, "y": [y], "both": (x, y)}
    mlir_str = sync.computation_mlir(pytree_input)

    # Verify that the MLIR string contains the expected operations
    self.assertIn("stablehlo.add", mlir_str)


if __name__ == "__main__":
  absltest.main()
