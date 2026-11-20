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

from collections.abc import Collection, Iterator

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export.unflatten import _assign_attr
from torch.export.unflatten import _AttrKind
from torch_tpu._internal.compile import tpu_torch_compile


def _iter_tensor_get_attrs(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
    skip_targets: Collection[str] = (),
) -> Iterator[tuple[torch.fx.GraphModule, str, torch.Tensor]]:
  """Yields `(gm, attr_target, tensor)` for tensor `get_attr` targets."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    root = gm_or_graph
  elif isinstance(gm_or_graph.owning_module, torch.fx.GraphModule):
    root = gm_or_graph.owning_module
  else:
    return

  for gm in root.modules():
    if not isinstance(gm, torch.fx.GraphModule):
      continue
    unique_attr_targets = {
        str(node.target) for node in gm.graph.find_nodes(op="get_attr")
    }
    for attr_target in unique_attr_targets:
      if attr_target in skip_targets:
        continue
      # pylint: disable-next=protected-access
      target = torch.fx.graph_module._get_attr(gm, attr_target)
      if isinstance(target, torch.Tensor):
        yield gm, attr_target, target


def move_constants_to_cpu(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Moves device tensor `get_attr` attributes in `gm_or_graph` to CPU."""
  # Exit Fake mode since `.to("cpu")` may run under an active FakeTensorMode.
  with unset_fake_temporarily():
    for gm, attr_target, target in _iter_tensor_get_attrs(gm_or_graph):
      if target.device.type != "cpu":
        _assign_attr(target.to("cpu"), gm, attr_target, _AttrKind.CONSTANT)


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
    skip_targets: Collection[str] = (),
) -> None:
  """Marks embedded constants (get_attr node targets) in an FX graph so that they may be properly handled during MLIR generation.

  This pass finds all `get_attr` nodes that reference torch.Tensor attributes
  on the module and replaces them with a new target tensor that will generate a
  deferred op during graph traversal via EagerLikeFxInterpreter. This enables us
  to properly embed a ConstantOp in the MLIR graph.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
    skip_targets: `get_attr` targets to leave unmarked, e.g. constants that are
      consumed on CPU.
  """
  # pylint: disable=protected-access
  with unset_fake_temporarily():
    for gm, attr_target, target in _iter_tensor_get_attrs(
        gm_or_graph, skip_targets=skip_targets
    ):
      if not hasattr(gm, "_processed_constant_attrs"):
        gm._processed_constant_attrs = set()  # pyrefly: ignore[missing-attribute]
      if attr_target in gm._processed_constant_attrs:  # pyrefly: ignore[missing-attribute]
        continue
      if tpu_torch_compile.is_constant_tensor(target):
        gm._processed_constant_attrs.add(attr_target)  # pyrefly: ignore[missing-attribute]
        continue

      # The target will be placed on the device by AOT Autograd automatically,
      # transfer it back to CPU so that we can access the data.
      new_target = tpu_torch_compile.make_constant_tensor(target.to("cpu"))
      _assign_attr(new_target, gm, attr_target, _AttrKind.CONSTANT)
      gm._processed_constant_attrs.add(attr_target)  # pyrefly: ignore[missing-attribute]
