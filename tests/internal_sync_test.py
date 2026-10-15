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

from collections.abc import Iterator
import contextlib
import os
import re
import tempfile
import textwrap
from typing import TypeAlias

from absl.testing import absltest
import torch
from torch_tpu._internal import execution_mode
from torch_tpu._internal import sync
from torch_tpu._internal import testing as tt_testing

_EagerMode: TypeAlias = execution_mode._EagerMode


class InternalSyncTest(absltest.TestCase):  # ABSLTEST_OK=Internal sync test

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    self.old_eager_mode = execution_mode.eager_mode
    execution_mode.eager_mode = _EagerMode.INTERNAL_DEFER_ALL

  def tearDown(self):
    execution_mode.eager_mode = self.old_eager_mode
    super().tearDown()

  @contextlib.contextmanager
  def undeclared_outputs_dir(self) -> Iterator[str]:

    if "TEST_UNDECLARED_OUTPUTS_DIR" in os.environ:
      context = contextlib.nullcontext(None)
    else:
      context = tempfile.TemporaryDirectory()

    with context as possible_temp_dir:
      if possible_temp_dir is not None:
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = possible_temp_dir
        dir_set = True
      else:
        dir_set = False

      try:
        yield os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]
      finally:
        if dir_set:
          del os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]

  def test_sync_no_wait_tensor(self):
    x = torch.ones(10, device=torch.device("tpu"))
    y = torch.ones(10, device=torch.device("tpu"))
    z = x + y

    # Nothing is materialized or ready.
    for tensor in [x, y, z]:
      self.assertFalse(sync.is_materializing(tensor))
      self.assertFalse(sync.is_materialized(tensor))

    sync.synchronize(y, wait=False)

    # The entire stream begins materializing, but may not be ready yet.
    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materializing(y))
    self.assertTrue(sync.is_materializing(z))

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

    # The entire stream starts materializing up to the point in time where the
    # synchronize call was made, including all of x, y, and z.
    # We block until y is materialized, and x was dispatched before y, so both
    # x and y are materialized and ready.
    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materializing(y))
    self.assertTrue(sync.is_materialized(y))

    # z will have started materializing, but may not have finished yet.
    self.assertTrue(sync.is_materializing(z))

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

    # Everything is materialized, but may not be ready.
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
    expected_graphviz_string = textwrap.dedent("""\
        Graphviz string: (try pasting in http://graphviz/ to see the graph)
        digraph {
          // Vertices:
          0 [shape="box", label=" uint64[2] (materialized)"];
          1 [label="uniform_"];
          2 [shape="box", label="x: float32[2, 3]"];
          3 [label="sum.IntList_out"];
          4 [shape="box", label="z: float32[]"];
          5 [label="pow.Tensor_Scalar_out"];
          6 [shape="box", label="y: float32[2, 3]"];

          // Edges:
          0 -> 1
          1 -> 2
          2 -> 3
          3 -> 4
          2 -> 5
          5 -> 6
        }
        """)

    node_params, num_lines = self.extract_graphviz_invariants(
        expected_graphviz_string
    )
    s = sync.computation_graphviz(y, z)
    for node_param in node_params:
      self.assertRegex(s, node_param)
    self.assertLen(s.split("\n"), num_lines)

  def test_dump_computation_graphviz(self):
    x = torch.ones(2, 3, device=torch.device("tpu"))
    y = torch.ones(2, 3, device=torch.device("tpu"))
    z = x + y
    expected_graphviz_string = textwrap.dedent("""\
        Graphviz string: (try pasting in http://graphviz/ to see the graph)
        digraph {
          // Vertices:
          0 [label="ones"];
          1 [shape="box", label=" float32[2, 3]"];
          2 [label="ones"];
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
        """)
    with self.undeclared_outputs_dir() as output_dir:
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

  def test_sync_tensor_subclass_wrappers(self):
    class _LocalTensorSubclass(torch.Tensor):
      pass

    class _ElemTensorSubclass(torch.Tensor):
      pass

    x = torch.ones(10, device=torch.device("tpu")) + 1.0
    y = torch.ones(10, device=torch.device("tpu")) + 2.0

    # Create wrapper tensor subclasses with ._local_tensor (like DTensor) and
    # .elem (like FakeTensor) whose outer storage is on CPU.
    dt_wrapper = _LocalTensorSubclass(torch.empty(0))
    dt_wrapper._local_tensor = x

    elem_wrapper = _ElemTensorSubclass(torch.empty(0))
    elem_wrapper.elem = y

    self.assertFalse(sync.is_materializing(dt_wrapper))
    self.assertFalse(sync.is_materializing(elem_wrapper))

    # Single-tensor synchronize on wrapper subclass
    sync.synchronize(dt_wrapper, wait=False)

    # The entire stream is now materializing.
    self.assertTrue(sync.is_materializing(dt_wrapper))
    self.assertTrue(sync.is_materializing(x))
    self.assertTrue(sync.is_materializing(elem_wrapper))

    # Sequence synchronize on list of wrapper subclasses
    sync.synchronize([dt_wrapper, elem_wrapper], wait=True)
    self.assertTrue(sync.is_materialized(dt_wrapper))
    self.assertTrue(sync.is_materialized(elem_wrapper))
    self.assertTrue(sync.is_materialized(x))
    self.assertTrue(sync.is_materialized(y))

  def test_sync_sparse_coo_tensor(self):
    indices = torch.tensor([[0, 1], [1, 0]], device=torch.device("tpu"))
    values = torch.tensor([3.0, 4.0], device=torch.device("tpu")) + 1.0
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 2))

    # Single-tensor synchronize on sparse COO tensor
    sync.synchronize(sparse_tensor, wait=True)
    self.assertTrue(sync.is_materialized(sparse_tensor._indices()))
    self.assertTrue(sync.is_materialized(sparse_tensor._values()))

    # List synchronize on sparse COO tensor
    values2 = torch.tensor([5.0, 6.0], device=torch.device("tpu")) + 2.0
    sparse_tensor2 = torch.sparse_coo_tensor(indices, values2, (2, 2))
    sync.synchronize([sparse_tensor2], wait=True)
    self.assertTrue(sync.is_materialized(sparse_tensor2._values()))


if __name__ == "__main__":
  absltest.main()
