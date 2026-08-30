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

"""Reorders SymInt placeholders to the beginning of submodule signatures."""

from absl import logging
import torch
from torch_tpu._internal.compile.dynamic import sym_utils


def apply(gm: torch.fx.GraphModule) -> None:
  """Reorders SymInt placeholders to the beginning of submodule signatures.

  Args:
    gm: The split GraphModule to reorder SymInts in.
  """
  submod_call_nodes = {}
  for node in gm.graph.nodes:
    if node.op == "call_module":
      submod_call_nodes[node.target] = node

  for submod_name, submod in gm.named_children():
    if not isinstance(submod, torch.fx.GraphModule):
      continue

    call_site_node = submod_call_nodes.get(submod_name)
    if call_site_node is None:
      continue

    placeholders = [
        node for node in submod.graph.nodes if node.op == "placeholder"
    ]
    if not placeholders:
      continue

    assert len(placeholders) == len(
        call_site_node.args
    ), f"Mismatch between placeholders and args for {submod_name}"

    symint_pairs = []
    other_pairs = []
    for ph, arg in zip(placeholders, call_site_node.args):
      if sym_utils.is_symint_node(ph):
        symint_pairs.append((ph, arg))
      else:
        other_pairs.append((ph, arg))

    if not symint_pairs or not other_pairs:
      continue

    new_pairs = symint_pairs + other_pairs
    new_placeholders = [ph for ph, _ in new_pairs]

    if new_placeholders == placeholders:
      continue

    logging.debug(
        "Reordering SymInt placeholders for submodule %s", submod_name
    )

    first_node = next(iter(submod.graph.nodes))
    for ph in reversed(new_placeholders):
      first_node.prepend(ph)
      first_node = ph

    call_site_node.args = tuple(arg for _, arg in new_pairs)

    submod.graph.lint()
    submod.recompile()

  gm.graph.lint()
  gm.recompile()
