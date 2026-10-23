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

"""Unit tests for sink_get_attr_constants pass."""

from absl.testing import absltest
import torch
from torch import fx
from torch_tpu._internal.compile.fx_passes import sink_get_attr_constants
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph


class SinkGetAttrConstantsTest(seed_test_utils.RepeatableTest):

  def test_placeholder_arg_count_mismatch_raises_value_error(self):
    sub_graph = Graph()
    sub_x = sub_graph.placeholder("x")
    sub_c = sub_graph.placeholder("c")
    sub_out = sub_graph.call_function(torch.ops.aten.add.Tensor, (sub_x, sub_c))
    sub_graph.output(sub_out)
    submod = GraphModule(torch.nn.Module(), sub_graph)

    # Submodule has 2 placeholders but the call site only passes 1 argument.
    parent_graph = Graph()
    p_x = parent_graph.placeholder("x")
    call_submod = parent_graph.call_module("submod_0", (p_x,))
    parent_graph.output(call_submod)

    parent_root = torch.nn.Module()
    parent_root.submod_0 = submod
    parent_gm = GraphModule(parent_root, parent_graph)

    with self.assertRaises(ValueError):
      sink_get_attr_constants.apply(parent_gm)


if __name__ == "__main__":
  absltest.main()
