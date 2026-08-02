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

"""Unit tests for dynamic shape compilation FX graph transformations."""

import operator
import unittest
import torch
from torch import fx

# Register fallback definition for set_dimension_logical_size if C++ extension is unbuilt
if not hasattr(torch.ops, "tpu") or not hasattr(torch.ops.tpu, "set_dimension_logical_size"):
  _tpu_lib = torch.library.Library("tpu", "FRAGMENT")
  _tpu_lib.define("set_dimension_logical_size(Tensor self, int dim, Tensor size) -> Tensor")
  @torch.library.impl(_tpu_lib, "set_dimension_logical_size", "CompositeExplicitAutograd")
  def _set_dim_size_impl(self, dim, size):
    return self

from torch_tpu._internal.compile.dynamic.graph_transformations import (
    apply_dynamism_transformations,
    DetectSymIntUsagesPass,
)
from torch_tpu._internal.compile.dynamic.sym_shape_manager import SymShapeManager


class DynamicCompilationTest(unittest.TestCase):

  def test_mul_symint_lowering_pass(self):
    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    symint_s0 = shape_env.create_symintnode(
        shape_env.create_symbol(4, torch._dynamo.source.ConstantSource("s0")),
        hint=4,
    )
    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode(
        shape_env=shape_env
    )

    with fake_mode:
      x_fake = torch.empty((symint_s0, 16), dtype=torch.float32)

      graph = fx.Graph()
      # Place SymInt placeholder first so created size placeholders appear before tensor consumers
      p_s0 = graph.placeholder("s0")
      p_s0.meta["val"] = symint_s0

      p_x = graph.placeholder("x")
      p_x.meta["val"] = x_fake

      mul_node = graph.call_function(
          operator.mul, (symint_s0, 8)
      )
      mul_node.meta["val"] = symint_s0 * 8

      empty_fake = torch.empty((symint_s0 * 8, 16), dtype=torch.float32)
      empty_node = graph.call_function(
          torch.ops.aten.empty.memory_format, ((mul_node, 16),)
      )
      empty_node.meta["val"] = empty_fake

      graph.output(empty_node)
      gm = fx.GraphModule(torch.nn.Module(), graph)

    sym_shape_manager = SymShapeManager(
        graph_module=gm,
        example_inputs=[symint_s0, x_fake],
    )

    apply_dynamism_transformations(gm, sym_shape_manager)

    unhandled_usages = DetectSymIntUsagesPass()(gm)
    self.assertEqual(
        len(unhandled_usages),
        0,
        f"Expected 0 unhandled SymInt nodes, got: {unhandled_usages}",
    )


if __name__ == "__main__":
  unittest.main()
