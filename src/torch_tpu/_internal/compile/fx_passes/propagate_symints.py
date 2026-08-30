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

"""Propagates SymInt placeholders to submodules that need them."""

from absl import logging
import torch
from torch_tpu._internal.compile.dynamic import sym_utils


def apply(gm: torch.fx.GraphModule) -> None:
  """Propagates SymInt placeholders to submodules that need them.

  This is required because split_module does not propagate SymInt placeholders
  to submodules unless they are explicitly used in the submodule body.

  The pass detects SymInts that are used in the submodule body but are not yet
  propagated form the parent module.
  It then adds these SymInt placeholders to the submodule signature and
  updates the call site node in the parent module.

  Example:
    GraphModule before propagating SymInts:
      def forward(self, s0: "Sym(s95)", x: "f32[s95, 4]"):
          submod_0 = self.submod_0(x);  x = None
          return submod_0

      class submod_0(torch.nn.Module):
          def forward(self, x: "f32[s95, 4]"):
              add_tensor = torch.ops.aten.add.Tensor(x, x);  x = None
              return add_tensor

    GraphModule after propagating SymInts:
      def forward(self, s0: "Sym(s95)", x: "f32[s95, 4]"):
          submod_0 = self.submod_0(s0, x);  s0 = x = None
          return submod_0

      class submod_0(torch.nn.Module):
          def forward(self, sym_s95: "Sym(s95)", x: "f32[s95, 4]"):
              add_tensor = torch.ops.aten.add.Tensor(x, x);  x = None
              return add_tensor

  Args:
    gm: The split GraphModule to propagate SymInts in.
  """
  parent_symint_placeholders = {}
  submod_call_nodes = {}
  for node in gm.graph.nodes:
    if node.op == "placeholder" and sym_utils.is_symint_node(node):
      parent_symint_placeholders[str(node.meta["val"])] = node
    elif node.op == "call_module":
      submod_call_nodes[node.target] = node

  if not parent_symint_placeholders:
    return

  for submod_name, submod in gm.named_children():
    if not isinstance(submod, torch.fx.GraphModule):
      continue

    submod_symbols = set()
    for node in submod.graph.nodes:
      if node.op == "placeholder" and isinstance(
          node.meta.get("val"), torch.Tensor
      ):
        for dim in node.meta["val"].shape:
          if isinstance(dim, torch.SymInt) and bool(dim.node.expr.free_symbols):
            submod_symbols.add(str(dim))

    existing_submod_symbols = set()
    for node in submod.graph.nodes:
      if node.op == "placeholder" and sym_utils.is_symint_node(node):
        existing_submod_symbols.add(str(node.meta["val"]))

    missing_symbols = submod_symbols - existing_submod_symbols
    if not missing_symbols:
      continue

    logging.debug(
        "Propagating SymInts %s to submodule %s",
        missing_symbols,
        submod_name,
    )

    # Sort for determinism
    sorted_missing_symbols = sorted(missing_symbols)

    call_site_node = submod_call_nodes.get(submod_name)

    assert (
        call_site_node is not None
    ), f"Could not find call site for {submod_name}"

    for sym_str in sorted_missing_symbols:
      parent_node = parent_symint_placeholders.get(sym_str)
      if parent_node is None:
        logging.warning(
            "SymInt %s used in %s but not found in parent placeholders",
            sym_str,
            submod_name,
        )
        continue

      first_node = next(iter(submod.graph.nodes))
      with submod.graph.inserting_before(first_node):
        new_ph = submod.graph.placeholder(f"sym_{sym_str}")
        new_ph.meta["val"] = parent_node.meta["val"]

      call_site_node.args = (parent_node,) + call_site_node.args

    submod.graph.lint()
    submod.recompile()

  gm.graph.lint()
  gm.recompile()
