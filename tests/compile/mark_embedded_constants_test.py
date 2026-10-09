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

"""Unit tests for mark_embedded_constants pass."""

from unittest import mock
from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile import tpu_torch_compile
from torch_tpu._internal.compile.fx_passes import mark_embedded_constants
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class MarkEmbeddedConstantsTest(seed_test_utils.RepeatableTest):

  def test_mark_embedded_constants_single_pass(self):
    g = Graph()
    x = g.placeholder("x")
    c = g.get_attr("c_attr")
    out = g.call_function(torch.ops.aten.add.Tensor, (x, c))
    g.output(out)

    root = torch.nn.Module()
    root.c_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    gm = GraphModule(root, g)

    mark_embedded_constants.apply(gm)

    self.assertTrue(tpu_torch_compile.is_constant_tensor(gm.c_attr))
    self.assertIn("c_attr", gm._processed_constant_attrs)

  def test_mark_embedded_constants_idempotent(self):
    g = Graph()
    x = g.placeholder("x")
    c = g.get_attr("c_attr")
    out = g.call_function(torch.ops.aten.add.Tensor, (x, c))
    g.output(out)

    root = torch.nn.Module()
    root.c_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    gm = GraphModule(root, g)

    # First pass
    mark_embedded_constants.apply(gm)
    first_tensor = gm.c_attr
    self.assertTrue(tpu_torch_compile.is_constant_tensor(first_tensor))

    # Second pass on the same GraphModule
    with mock.patch.object(
        tpu_torch_compile,
        "make_constant_tensor",
        wraps=tpu_torch_compile.make_constant_tensor,
    ) as mock_make:
      mark_embedded_constants.apply(gm)
      mock_make.assert_not_called()

    self.assertIs(gm.c_attr, first_tensor)

  def test_mark_embedded_constants_idempotent_new_graph_module(self):
    """Simulates submodule extraction or deepcopy where _processed_constant_attrs is missing."""
    g = Graph()
    x = g.placeholder("x")
    c = g.get_attr("c_attr")
    out = g.call_function(torch.ops.aten.add.Tensor, (x, c))
    g.output(out)

    root = torch.nn.Module()
    root.c_attr = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    gm = GraphModule(root, g)

    # First pass marks it
    mark_embedded_constants.apply(gm)
    first_tensor = gm.c_attr
    self.assertTrue(tpu_torch_compile.is_constant_tensor(first_tensor))

    # Clear _processed_constant_attrs or create a new GraphModule with the same marked tensor
    sub_g = Graph()
    sub_x = sub_g.placeholder("x")
    sub_c = sub_g.get_attr("c_attr")
    sub_g.output(sub_g.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c)))
    sub_root = torch.nn.Module()
    sub_root.c_attr = first_tensor
    sub_gm = GraphModule(sub_root, sub_g)

    # Verify _processed_constant_attrs is not present on sub_gm
    self.assertFalse(hasattr(sub_gm, "_processed_constant_attrs"))

    # Apply mark_embedded_constants again: it should identify the constant tensor
    # and skip calling make_constant_tensor or .to('cpu')
    with mock.patch.object(
        tpu_torch_compile,
        "make_constant_tensor",
        wraps=tpu_torch_compile.make_constant_tensor,
    ) as mock_make:
      mark_embedded_constants.apply(sub_gm)
      mock_make.assert_not_called()

    self.assertIs(sub_gm.c_attr, first_tensor)
    self.assertIn("c_attr", sub_gm._processed_constant_attrs)

  def test_mark_embedded_constants_skips_cpu_constants(self):
    g = Graph()
    x = g.placeholder("x")
    c = g.get_attr("cpu_c_attr")
    c.meta["val"] = torch.empty((1,), device="cpu")
    out = g.call_function(torch.ops.aten.add.Tensor, (x, c))
    g.output(out)

    root = torch.nn.Module()
    orig_tensor = torch.tensor([1.0], dtype=torch.float32)
    root.cpu_c_attr = orig_tensor
    gm = GraphModule(root, g)

    with mock.patch.object(
        tpu_torch_compile,
        "make_constant_tensor",
        wraps=tpu_torch_compile.make_constant_tensor,
    ) as mock_make:
      mark_embedded_constants.apply(gm)
      mock_make.assert_not_called()

    self.assertIs(gm.cpu_c_attr, orig_tensor)
    self.assertFalse(tpu_torch_compile.is_constant_tensor(gm.cpu_c_attr))


if __name__ == "__main__":
  absltest.main()
