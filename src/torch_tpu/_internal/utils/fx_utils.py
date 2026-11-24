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

"""Utility helpers for inspecting PyTorch FX graphs and GraphModules."""

from collections.abc import Collection, Iterator

import torch


def iter_tensor_get_attrs(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
    skip_targets: Collection[str] = (),
) -> Iterator[tuple[torch.fx.GraphModule, str, torch.Tensor]]:
  """Yields `(gm, attr_target, tensor)` for unique tensor `get_attr` targets.

  Traverses `gm_or_graph` (or its owning `GraphModule`) and any nested
  `GraphModule`s, yielding each unique `get_attr` target in graph order whose
  resolved attribute is a `torch.Tensor`.

  Args:
    gm_or_graph: The root FX `GraphModule` or `Graph` to inspect.
    skip_targets: `get_attr` target names to skip.

  Yields:
    Tuples of `(gm, attr_target, target_tensor)`.
  """
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    root = gm_or_graph
  elif isinstance(gm_or_graph.owning_module, torch.fx.GraphModule):
    root = gm_or_graph.owning_module
  else:
    return

  for gm in root.modules():
    if not isinstance(gm, torch.fx.GraphModule):
      continue
    unique_attr_targets = dict.fromkeys(
        str(node.target) for node in gm.graph.find_nodes(op="get_attr")
    )
    for attr_target in unique_attr_targets:
      if attr_target in skip_targets:
        continue
      # pylint: disable-next=protected-access
      target = torch.fx.graph_module._get_attr(gm, attr_target)
      if isinstance(target, torch.Tensor):
        yield gm, attr_target, target
