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

"""Marks constants in the FX graph in preparation for MLIR serialization."""

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export.unflatten import _assign_attr
from torch.export.unflatten import _AttrKind
from torch_tpu._internal.compile import tpu_torch_compile


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Marks embedded constants (get_attr node targets) in an FX graph so that they may be properly handled during MLIR generation.

  This pass finds all `get_attr` nodes that reference torch.Tensor attributes
  on the module and replaces them with a new target tensor that will generate a
  deferred op during graph traversal via EagerLikeFxInterpreter. This enables us
  to properly embed a ConstantOp in the MLIR graph.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
  """
  # pylint: disable=protected-access
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    gm = gm_or_graph
    graph = gm_or_graph.graph
  else:
    gm = gm_or_graph.owning_module
    graph = gm_or_graph

  if not hasattr(gm, "_processed_constant_attrs"):
    gm._processed_constant_attrs = set()

  # Exit Fake mode since we're creating new tensors.
  with unset_fake_temporarily():
    unique_attr_targets = {
        node.target for node in graph.find_nodes(op="get_attr")
    }

    for attr_target in unique_attr_targets:
      if attr_target in gm._processed_constant_attrs:
        continue

      # pylint: disable-next=protected-access
      target = torch.fx.graph_module._get_attr(gm, attr_target)  # pyrefly: ignore[bad-argument-type]
      if not isinstance(target, torch.Tensor):
        continue

      # The target will be placed on the device by AOT Autograd automatically,
      # transfer it back to CPU so that we can access the data.
      new_target = tpu_torch_compile.make_constant_tensor(target.to("cpu"))
      # pylint: disable-next=protected-access
      _assign_attr(new_target, gm, attr_target, _AttrKind.CONSTANT)  # pyrefly: ignore[bad-argument-type]
      gm._processed_constant_attrs.add(attr_target)
