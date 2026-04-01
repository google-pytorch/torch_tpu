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

from absl.testing import absltest
import torch
from torch_tpu import api
from torch_tpu._internal import sync
from torch_tpu._internal.utils import utils


class SyncTest(absltest.TestCase):

  def test_sync_no_wait_tensor(self):
    x = torch.ones(10, device=api.tpu_device())
    y = torch.ones(10, device=api.tpu_device())
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materialized(tensor))
      self.assertFalse(sync.is_ready(tensor))

    sync.synchronize(z, wait=False)

    # x and y were not materialized.
    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))

    # z was materialized, but may or may not be ready.
    self.assertTrue(sync.is_materialized(z))

  def test_sync_no_wait_list(self):
    x = torch.ones(10, device=api.tpu_device())
    y = torch.ones(10, device=api.tpu_device())
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materialized(tensor))
      self.assertFalse(sync.is_ready(tensor))

    sync.synchronize([x, y, z], wait=False)

    # Everything is materialized, but may or may not be ready.
    for tensor in [x, y, z]:
      self.assertTrue(sync.is_materialized(tensor))

  def test_sync_and_wait_tensor(self):
    x = torch.ones(10, device=api.tpu_device())
    y = torch.ones(10, device=api.tpu_device())
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materialized(tensor))
      self.assertFalse(sync.is_ready(tensor))

    sync.synchronize(z, wait=True)

    # x and y were not materialized.
    self.assertFalse(sync.is_materialized(x))
    self.assertFalse(sync.is_materialized(y))

    # z was materialized and is ready.
    self.assertTrue(sync.is_materialized(z))
    self.assertTrue(sync.is_ready(z))

  def test_sync_and_wait_list(self):
    x = torch.ones(10, device=api.tpu_device())
    y = torch.ones(10, device=api.tpu_device())
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materialized(tensor))
      self.assertFalse(sync.is_ready(tensor))

    sync.synchronize([x, y, z], wait=True)

    # Everything is materialized and ready.
    for tensor in [x, y, z]:
      self.assertTrue(sync.is_materialized(tensor))
      self.assertTrue(sync.is_ready(tensor))

  def test_sync_no_wait_all(self):
    x = torch.ones(10, device=api.tpu_device())
    y = torch.ones(11, device=api.tpu_device())
    z = torch.ones(12, device=api.tpu_device())

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materialized(tensor))
      self.assertFalse(sync.is_ready(tensor))

    sync.synchronize(wait=False)

    # Everything is materialized, but may or may not be ready.
    for tensor in [x, y, z]:
      self.assertTrue(sync.is_materialized(tensor))

  def test_sync_and_wait_all(self):
    x = torch.ones(10, device=api.tpu_device())
    y = torch.ones(11, device=api.tpu_device())
    z = torch.ones(12, device=api.tpu_device())

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materialized(tensor))
      self.assertFalse(sync.is_ready(tensor))

    sync.synchronize(wait=True)

    # Everything is materialized and ready.
    for tensor in [x, y, z]:
      self.assertTrue(sync.is_materialized(tensor))
      self.assertTrue(sync.is_ready(tensor))

  def test_host_to_device_is_materialized(self):
    x = torch.ones(128, device="cpu").to(api.tpu_device())

    self.assertTrue(sync.is_materialized(x))

    sync.synchronize(x, wait=True)

    self.assertTrue(sync.is_ready(x))

  def test_zero_size_always_ready(self):
    x = torch.ones(1024, 0, 1024, device=api.tpu_device())
    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_ready(x))

  def test_is_materialized_not_on_tpu(self):
    x = torch.ones(10, device=torch.device("cpu"))
    with self.assertRaisesRegex(
        RuntimeError,
        "tensor is not on the PrivateUse1 device",
    ):
      sync.is_materialized(x)

  def test_is_ready_not_on_tpu(self):
    x = torch.ones(10, device=torch.device("cpu"))
    with self.assertRaisesRegex(
        RuntimeError,
        "tensor is not on the PrivateUse1 device",
    ):
      sync.is_ready(x)

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
    x = torch.rand(2, 3, device=api.tpu_device())
    y = x**2
    z = x.sum()
    expected_graphviz_string = """Graphviz string: (try pasting in http://graphviz/ to see the graph)
digraph {
  // Vertices:
  0 [shape="box", label=" float32[]"];
  1 [label="fill_.Scalar"];
  2 [shape="box", label=" uint64[2]"];
  3 [label="set_seed"];
  4 [shape="box", label=" uint64[2]"];
  5 [label="uniform_"];
  6 [shape="box", label=" uint64[2]"];
  7 [shape="box", label="x: float32[2, 3]"];
  8 [label="sum"];
  9 [shape="box", label="z: float32[]"];
  10 [label="pow.out"];
  11 [shape="box", label="y: float32[2, 3]"];

  // Edges:
  1 -> 2
  2 -> 3
  3 -> 4
  4 -> 5
  5 -> 6
  5 -> 7
  7 -> 8
  8 -> 9
  7 -> 10
  0 -> 10
  10 -> 11
}
"""

    node_params, num_lines = self.extract_graphviz_invariants(
        expected_graphviz_string
    )
    s = sync.computation_graphviz(y, z)
    for node_param in node_params:
      self.assertRegex(s, node_param)
    self.assertLen(s.split("\n"), num_lines)

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
    # Keep all tensors alive to prevent stale heuristic from materializing them.
    # Create materialized leaf inputs by doing a host to device copy.
    x_ones = torch.ones(2, 3, device="cpu").to(api.tpu_device())
    y_ones = torch.ones(3, 4, device="cpu").to(api.tpu_device())
    z_ones = torch.ones(2, 4, device="cpu").to(api.tpu_device())
    two = torch.tensor(2, device="cpu").to(api.tpu_device())
    three = torch.tensor(3, device="cpu").to(api.tpu_device())
    four = torch.tensor(4, device="cpu").to(api.tpu_device())

    # Create a graph of deferred operations.
    x = x_ones * two
    y = y_ones * three
    z = z_ones * four
    x_times_y = x @ y
    w = x_times_y + z

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
    x = torch.ones(2, 3, device=api.tpu_device())
    y = torch.ones(2, 3, device=api.tpu_device())
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
    x = torch.ones(2, 3, device=api.tpu_device())
    y = x + 1.0

    # Passing a complex pytree (dict, list, tuple) to computation_mlir
    pytree_input = {"x": x, "y": [y], "both": (x, y)}
    mlir_str = sync.computation_mlir(pytree_input)

    # Verify that the MLIR string contains the expected operations
    self.assertIn("stablehlo.add", mlir_str)

  def test_sync_cpu_tensor_non_blocking(self):
    x = torch.ones(10, device=api.tpu_device())
    y = x + 1

    cpu_tensor = torch.empty(10, pin_memory=True)
    cpu_tensor.copy_(y, non_blocking=True)

    sync.synchronize(cpu_tensor, wait=True)

    utils.assert_close(cpu_tensor, torch.ones(10) * 2)

  def test_sync_tpu_and_cpu_pinned_tensor(self):
    x = torch.ones(10, device=api.tpu_device())
    y = x + 1

    cpu_tensor = torch.empty(10, pin_memory=True)
    cpu_tensor.copy_(y, non_blocking=True)

    sync.synchronize([y, cpu_tensor], wait=True)

    utils.assert_close(cpu_tensor, torch.ones(10) * 2)

  def test_sync_two_cpu_pinned_tensors(self):
    x1 = torch.ones(10, device=api.tpu_device())
    y1 = x1 + 1
    x2 = torch.ones(10, device=api.tpu_device())
    y2 = x2 + 2

    cpu_tensor1 = torch.empty(10, pin_memory=True)
    cpu_tensor2 = torch.empty(10, pin_memory=True)

    cpu_tensor1.copy_(y1, non_blocking=True)
    cpu_tensor2.copy_(y2, non_blocking=True)

    sync.synchronize([cpu_tensor1, cpu_tensor2], wait=True)

    utils.assert_close(cpu_tensor1, torch.ones(10) * 2)
    utils.assert_close(cpu_tensor2, torch.ones(10) * 3)

  def test_sync_non_pinned_cpu_tensor_fails(self):
    cpu_tensor = torch.ones(10)  # Not pinned

    with self.assertRaisesRegex(RuntimeError, "Tensor is not on pinned memory"):
      sync.synchronize(cpu_tensor, wait=True)


if __name__ == "__main__":
  absltest.main()
