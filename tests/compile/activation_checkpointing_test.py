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

from collections.abc import Callable, Sequence
import copy
import glob
import operator
import os
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._functorch import aot_autograd as functorch_aot
from torch.utils import checkpoint
import torch_tpu
from torch_tpu._internal.compile.debug import TpuCompileDebug
from torch_tpu._internal.compile.fx_passes import mark_activation_checkpoints
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


def _add_remat_node(
    graph: fx.Graph,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    ac_graph_id: int,
    name: str,
) -> fx.Node:
  node = graph.call_function(target, args, name=name)
  node.meta["ac_graph_id"] = ac_graph_id
  node.meta["recompute"] = checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
  node.meta["autograd_backward"] = True
  node.meta["val"] = torch.empty(4, 8)
  return node


def _add_bwd_node(
    graph: fx.Graph,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    name: str,
) -> fx.Node:
  node = graph.call_function(target, args, name=name)
  node.meta["autograd_backward"] = True
  node.meta["val"] = torch.empty(4, 8)
  return node


def _first_non_getitem_after(barrier: fx.Node) -> fx.Node:
  curr = barrier.next
  while (
      curr.op == "call_function"
      and curr.target == operator.getitem
      and curr.args[0] is barrier
  ):
    curr = curr.next
  return curr


def _get_barriers(gm_or_graph: fx.GraphModule | fx.Graph) -> list[fx.Node]:
  graph = (
      gm_or_graph.graph
      if isinstance(gm_or_graph, fx.GraphModule)
      else gm_or_graph
  )
  return [
      n
      for n in graph.nodes
      if n.op == "call_function"
      and n.target == torch.ops.tpu._optimization_barrier.default  # pylint: disable=protected-access
  ]


def _build_two_layer_graph(
    *, include_forward: bool = False
) -> tuple[fx.GraphModule, dict[str, fx.Node]]:
  """Builds a 2-layer checkpointed MLP backward (or joint fwd+bwd) FX graph."""
  graph = fx.Graph()
  nodes: dict[str, fx.Node] = {}
  sample = torch.empty(4, 8)
  mm = torch.ops.aten.mm.default
  gelu = torch.ops.aten.gelu.default
  mul = torch.ops.aten.mul.Tensor

  for name in ("x_1", "W1_1", "W2_1", "W1_2", "W2_2"):
    nodes[name] = graph.placeholder(name)
    nodes[name].meta["val"] = sample

  if include_forward:
    nodes["loss_seed"] = graph.placeholder("loss_seed")
    nodes["loss_seed"].meta["val"] = torch.tensor(1.0)

    def add_fwd_node(target, args, ac_graph_id, name):
      node = graph.call_function(target, args, name=name)
      node.meta["ac_graph_id"] = ac_graph_id
      node.meta["recompute"] = checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
      node.meta["autograd_backward"] = False
      node.meta["val"] = sample
      nodes[name] = node
      return node

    a_1_fwd = add_fwd_node(mm, (nodes["x_1"], nodes["W1_1"]), 0, "a_1_fwd")
    b_1_fwd = add_fwd_node(gelu, (a_1_fwd,), 0, "b_1_fwd")
    add_fwd_node(mm, (b_1_fwd, nodes["W2_1"]), 0, "x_2")

    a_2_fwd = add_fwd_node(mm, (nodes["x_2"], nodes["W1_2"]), 1, "a_2_fwd")
    b_2_fwd = add_fwd_node(gelu, (a_2_fwd,), 1, "b_2_fwd")
    y_2 = add_fwd_node(mm, (b_2_fwd, nodes["W2_2"]), 1, "y_2")
    nodes["dY_2"] = _add_bwd_node(graph, mul, (y_2, nodes["loss_seed"]), "dY_2")
  else:
    for name in ("x_2", "dY_2"):
      nodes[name] = graph.placeholder(name)
      nodes[name].meta["val"] = sample

  # Backward Region 1 (ac_graph_id = 1)
  nodes["a_2_remat"] = _add_remat_node(
      graph, mm, (nodes["x_2"], nodes["W1_2"]), 1, "a_2_remat"
  )
  nodes["a_2_remat"].meta["stack_trace"] = "layer_1_trace"
  nodes["a_2_remat"].meta["nn_module_stack"] = {"L1": ("L1", torch.nn.Linear)}
  nodes["b_2_remat"] = _add_remat_node(
      graph, gelu, (nodes["a_2_remat"],), 1, "b_2_remat"
  )
  nodes["dW2_2"] = _add_bwd_node(
      graph, mm, (nodes["b_2_remat"], nodes["dY_2"]), "dW2_2"
  )
  nodes["dB_2"] = _add_bwd_node(
      graph, mm, (nodes["dY_2"], nodes["W2_2"]), "dB_2"
  )
  nodes["dA_2"] = _add_bwd_node(
      graph, mul, (nodes["dB_2"], nodes["a_2_remat"]), "dA_2"
  )
  nodes["dW1_2"] = _add_bwd_node(
      graph, mm, (nodes["x_2"], nodes["dA_2"]), "dW1_2"
  )
  nodes["dY_1"] = _add_bwd_node(
      graph, mm, (nodes["dA_2"], nodes["W1_2"]), "dY_1"
  )

  # Backward Region 0 (ac_graph_id = 0)
  nodes["a_1_remat"] = _add_remat_node(
      graph, mm, (nodes["x_1"], nodes["W1_1"]), 0, "a_1_remat"
  )
  nodes["b_1_remat"] = _add_remat_node(
      graph, gelu, (nodes["a_1_remat"],), 0, "b_1_remat"
  )
  nodes["dW2_1"] = _add_bwd_node(
      graph, mm, (nodes["b_1_remat"], nodes["dY_1"]), "dW2_1"
  )
  nodes["dB_1"] = _add_bwd_node(
      graph, mm, (nodes["dY_1"], nodes["W2_1"]), "dB_1"
  )
  nodes["dA_1"] = _add_bwd_node(
      graph, mul, (nodes["dB_1"], nodes["a_1_remat"]), "dA_1"
  )
  nodes["dW1_1"] = _add_bwd_node(
      graph, mm, (nodes["x_1"], nodes["dA_1"]), "dW1_1"
  )
  nodes["dX_1"] = _add_bwd_node(
      graph, mm, (nodes["dA_1"], nodes["W1_1"]), "dX_1"
  )

  graph.output((
      nodes["dX_1"],
      nodes["dW1_1"],
      nodes["dW2_1"],
      nodes["dW1_2"],
      nodes["dW2_2"],
  ))
  return fx.GraphModule(torch.nn.Module(), graph), nodes


def _sac_policy_fn(ctx, op, *args, **kwargs):
  del ctx, args, kwargs
  if op in (torch.ops.aten.addmm.default, torch.ops.aten.mm.default):
    return checkpoint.CheckpointPolicy.MUST_SAVE
  return checkpoint.CheckpointPolicy.PREFER_RECOMPUTE


def _sac_context_fn():
  return checkpoint.create_selective_checkpoint_contexts(_sac_policy_fn)


class _TwoLayerCheckpointedMLP(torch.nn.Module):

  def __init__(self, use_checkpoint: bool = True):
    super().__init__()
    self.use_checkpoint = use_checkpoint
    self.layers = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)
        )
        for _ in range(2)
    ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      if self.use_checkpoint:
        x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
      else:
        x = layer(x)
    return x


class _SharedWeightCheckpointedMLP(torch.nn.Module):

  def __init__(self, use_checkpoint: bool = True):
    super().__init__()
    self.use_checkpoint = use_checkpoint
    self.shared_layer = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for _ in range(2):
      if self.use_checkpoint:
        x = torch.utils.checkpoint.checkpoint(
            self.shared_layer, x, use_reentrant=False
        )
      else:
        x = self.shared_layer(x)
    return x


class _SACBlock(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.ln = torch.nn.LayerNorm(8)
    self.fc1 = torch.nn.Linear(8, 8)
    self.act = torch.nn.GELU(approximate="none")
    self.fc2 = torch.nn.Linear(8, 8)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc2(self.act(self.fc1(self.ln(x))))


class _TwoLayerSACModel(torch.nn.Module):

  def __init__(self, use_sac: bool):
    super().__init__()
    self.block1 = _SACBlock()
    self.block2 = _SACBlock()
    self.use_sac = use_sac

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not self.use_sac:
      return self.block2(self.block1(x))

    for block in (self.block1, self.block2):
      x = checkpoint.checkpoint(
          block,
          x,
          use_reentrant=False,
          context_fn=_sac_context_fn,
      )
    return x


class ActivationCheckpointingTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    self._dump_dir = self.create_tempdir().full_path
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {
                "TEST_UNDECLARED_OUTPUTS_DIR": self._dump_dir,
                "XLA_FLAGS": (
                    "--xla_dump_to=sponge --xla_dump_hlo_as_text"
                    " --xla_dump_hlo_pass_re=cse"
                ),
            },
        )
    )
    super().setUp()

  def _assert_hlo_cse_idempotent(
      self,
      expected_barriers: int,
      expected_erfs: int,
      expected_dots: int,
      *,
      min_erfs_after_cse: int | None = None,
  ) -> None:
    all_module_before = sorted(
        glob.glob(
            os.path.join(self._dump_dir, "module_*_split*_*.before_cse.txt")
        )
    )
    self.assertNotEmpty(all_module_before)
    module_prefix = os.path.basename(all_module_before[-1]).split(".")[0]

    before_files = sorted(
        glob.glob(
            os.path.join(self._dump_dir, f"{module_prefix}.*.before_cse.txt")
        )
    )
    after_files = sorted(
        glob.glob(
            os.path.join(self._dump_dir, f"{module_prefix}.*.after_cse.*.txt")
        )
    )
    self.assertNotEmpty(before_files)
    self.assertEqual(len(before_files), len(after_files))

    with open(before_files[0], "r") as f:
      first_entry = f.read().split("ENTRY ", maxsplit=1)[1]
    self.assertEqual(first_entry.count("opt-barrier("), expected_barriers)
    self.assertEqual(first_entry.count("erf("), expected_erfs)
    self.assertEqual(first_entry.count("dot("), expected_dots)

    invariant_tokens = (
        ("opt-barrier(", "dot(", "rsqrt(")
        if min_erfs_after_cse is not None
        else ("opt-barrier(", "erf(", "dot(", "rsqrt(")
    )
    for before_path, after_path in zip(before_files, after_files, strict=True):
      with open(before_path, "r") as f:
        before_entry = f.read().split("ENTRY ", maxsplit=1)[1]
      with open(after_path, "r") as f:
        after_entry = f.read().split("ENTRY ", maxsplit=1)[1]
      for op_token in invariant_tokens:
        self.assertEqual(
            before_entry.count(op_token),
            after_entry.count(op_token),
            f"HloCSE changed {op_token} count between"
            f" {os.path.basename(before_path)} and"
            f" {os.path.basename(after_path)}",
        )
      if min_erfs_after_cse is not None and before_entry.count("erf(") > 0:
        self.assertGreaterEqual(after_entry.count("erf("), min_erfs_after_cse)

  def _run_and_verify_tpu_model(
      self,
      model: torch.nn.Module,
      ref_model: torch.nn.Module,
      *,
      fuse_step: bool,
      input_shape: Sequence[int] = (4, 8),
  ) -> TpuCompileDebug:
    device = torch.device("tpu")
    model = model.to(device).train()
    ref_model = ref_model.to(device).train()
    ref_model.load_state_dict(copy.deepcopy(model.state_dict()))

    criterion = torch.nn.MSELoss()
    debugs: list[TpuCompileDebug] = []
    options = {"serializable": False, "debug_callback": debugs.append}

    x = torch.randn(*input_shape, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    target = torch.randn(*input_shape, device=device)

    ref_outputs = ref_model(ref_x)
    ref_loss = criterion(ref_outputs, target)
    ref_loss.backward()

    if fuse_step:

      def train_step(inp, tgt):
        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        return loss.detach()

      compiled_step = torch.compile(train_step, backend="tpu", options=options)
      loss_out = compiled_step(x, target)
      test_utils.assert_close(loss_out, ref_loss.detach())
    else:
      compiled_model = torch.compile(model, backend="tpu", options=options)
      outputs = compiled_model(x)
      loss = criterion(outputs, target)
      loss.backward()
      test_utils.assert_close(outputs, ref_outputs)

    test_utils.assert_close(x.grad, ref_x.grad)
    for p, ref_p in zip(
        model.parameters(), ref_model.parameters(), strict=True
    ):
      test_utils.assert_close(p.grad, ref_p.grad)

    self.assertLen(debugs, 1)
    return debugs[0]

  def test_graph_without_backward_checkpoints_remains_unmodified(self):
    for autograd_backward, with_ac_tags in (
        (False, False),
        (False, True),
        (True, False),
    ):
      graph = fx.Graph()
      x = graph.placeholder("x")
      w = graph.placeholder("w")
      out = graph.call_function(torch.ops.aten.mm.default, (x, w))
      out.meta["autograd_backward"] = autograd_backward
      if with_ac_tags:
        out.meta["ac_graph_id"] = 0
        out.meta["recompute"] = checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
      graph.output(out)
      gm = fx.GraphModule(torch.nn.Module(), graph)

      nodes_before = [(n.op, n.target, n.args) for n in gm.graph.nodes]
      code_before = gm.code
      mark_activation_checkpoints.apply(gm)
      self.assertEqual(
          nodes_before, [(n.op, n.target, n.args) for n in gm.graph.nodes]
      )
      self.assertEqual(code_before, gm.code)

  def test_two_layer_variadic_barriers_and_idempotency(self):
    for include_forward in (False, True):
      gm, nodes = _build_two_layer_graph(include_forward=include_forward)
      mark_activation_checkpoints.apply(gm)

      barriers = _get_barriers(gm)
      self.assertLen(barriers, 2)
      self.assertIs(_first_non_getitem_after(barriers[0]), nodes["a_2_remat"])
      self.assertIs(_first_non_getitem_after(barriers[1]), nodes["a_1_remat"])

      if include_forward:
        # Forward checkpointed ops remain untouched with un-barriered inputs.
        self.assertEqual(nodes["a_1_fwd"].args, (nodes["x_1"], nodes["W1_1"]))
        self.assertEqual(nodes["x_2"].args, (nodes["b_1_fwd"], nodes["W2_1"]))
        self.assertEqual(nodes["a_2_fwd"].args, (nodes["x_2"], nodes["W1_2"]))
        self.assertEqual(nodes["y_2"].args, (nodes["b_2_fwd"], nodes["W2_2"]))

      # Each barrier binds its region's saved inputs/weights and incoming
      # upstream gradient in deterministic consumer order (excluding scalar
      # `loss_seed` and upstream weight gradients `dW2_2`/`dW1_2`).
      self.assertEqual(
          list(barriers[0].args[0]),
          [nodes["x_2"], nodes["W1_2"], nodes["dY_2"], nodes["W2_2"]],
      )
      self.assertEqual(
          list(barriers[1].args[0]),
          [nodes["x_1"], nodes["W1_1"], nodes["dY_1"], nodes["W2_1"]],
      )

      # All backward remat and multi-hop gradient ops consume barrier outputs.
      for op_name in ("a_2_remat", "dW2_2", "dB_2", "dW1_2", "dY_1"):
        self.assertTrue(
            any(
                inp.args and inp.args[0] is barriers[0]
                for inp in nodes[op_name].all_input_nodes
            )
        )
      for op_name in ("a_1_remat", "dW2_1", "dB_1", "dW1_1", "dX_1"):
        self.assertTrue(
            any(
                inp.args and inp.args[0] is barriers[1]
                for inp in nodes[op_name].all_input_nodes
            )
        )

      # Metadata from `entry_node` is copied to the barrier and its `getitem`s.
      for node in [barriers[0], *barriers[0].users]:
        self.assertIn("val", node.meta)
        self.assertTrue(node.meta.get("autograd_backward", False))
        self.assertEqual(node.meta.get("stack_trace"), "layer_1_trace")
        self.assertEqual(
            node.meta.get("nn_module_stack"),
            {"L1": ("L1", torch.nn.Linear)},
        )

      # Applying the pass a second time is a no-op (idempotent).
      nodes_after_first = [(n.op, n.target, n.args) for n in gm.graph.nodes]
      mark_activation_checkpoints.apply(gm)
      self.assertEqual(
          nodes_after_first, [(n.op, n.target, n.args) for n in gm.graph.nodes]
      )

  def test_tuple_getitems_and_barrier_input_filtering(self):
    gm, nodes = _build_two_layer_graph(include_forward=False)
    graph = gm.graph
    sample = torch.empty(4, 8)

    # 1. Insert pre-region collective (`all_gather` + `wait_tensor`), nested
    #    tuple op (`fwd_tuple`) with both a non-straddling `fwd_getitem` and a
    #    straddling nested `inner_tup -> leaf_tensor` after `a_2_remat`, plus
    #    scalar (`ndim == 0`), 0-input factory (`ones`), and `get_attr` inputs.
    with graph.inserting_before(nodes["a_2_remat"]):
      ag_w1_2 = _add_bwd_node(
          graph,
          torch.ops._c10d_functional.all_gather_into_tensor.default,  # pylint: disable=protected-access
          (nodes["W1_2"], 2, "group"),
          "ag_W1_2",
      )
      wait_ag_w1_2 = _add_bwd_node(
          graph,
          torch.ops._c10d_functional.wait_tensor.default,  # pylint: disable=protected-access
          (ag_w1_2,),
          "wait_ag_W1_2",
      )
      fwd_tuple = graph.call_function(
          torch.ops.aten.split.Tensor, (nodes["dY_2"], 4), name="fwd_tuple"
      )
      fwd_tuple.meta["val"] = (sample, (sample, None))
      fwd_other = graph.call_function(
          torch.ops.aten.mm.default,
          (nodes["x_2"], nodes["W2_2"]),
          name="fwd_other",
      )
      fwd_other.meta["val"] = sample
      fwd_getitem = graph.call_function(
          operator.getitem, (fwd_tuple, 0), name="fwd_getitem"
      )
      fwd_getitem.meta["val"] = sample
      scalar_seed = _add_bwd_node(
          graph, torch.ops.aten.sum.default, (nodes["dY_2"],), "scalar_seed"
      )
      scalar_seed.meta["val"] = torch.empty(())
      factory_ones = _add_bwd_node(
          graph, torch.ops.aten.ones.default, ([4, 8],), "factory_ones"
      )
      gm.register_buffer("const_buf", sample)
      attr_const = graph.get_attr("const_buf")
      attr_const.meta["val"] = sample

    nodes["a_2_remat"].replace_input_with(nodes["W1_2"], wait_ag_w1_2)
    nodes["dY_1"].replace_input_with(nodes["W1_2"], wait_ag_w1_2)

    with graph.inserting_before(nodes["dW2_2"]):
      b2_t = _add_bwd_node(
          graph, torch.ops.aten.t.default, (nodes["b_2_remat"],), "b2_t"
      )
      inner_tup = graph.call_function(
          operator.getitem, (fwd_tuple, 1), name="inner_tup"
      )
      inner_tup.meta["val"] = (sample, None)
      leaf_tensor = graph.call_function(
          operator.getitem, (inner_tup, 0), name="leaf_tensor"
      )
      leaf_tensor.meta["val"] = sample
      leaf_none = graph.call_function(
          operator.getitem, (inner_tup, 1), name="leaf_none"
      )
      leaf_none.meta["val"] = None
      scaled = _add_bwd_node(
          graph,
          torch.ops.aten.add.Tensor,
          (leaf_tensor, scalar_seed),
          "scaled",
      )
      with_factory = _add_bwd_node(
          graph,
          torch.ops.aten.add.Tensor,
          (scaled, factory_ones),
          "with_factory",
      )
      with_attr = _add_bwd_node(
          graph,
          torch.ops.aten.add.Tensor,
          (with_factory, attr_const),
          "with_attr",
      )

    nodes["dW2_2"].replace_input_with(nodes["b_2_remat"], b2_t)
    nodes["dW2_2"].replace_input_with(nodes["dY_2"], with_attr)

    # 2. Insert a post-region weight-gradient `reduce_scatter` before Layer 1.
    with graph.inserting_before(nodes["a_1_remat"]):
      rs = _add_bwd_node(
          graph,
          torch.ops._c10d_functional.reduce_scatter_tensor.default,  # pylint: disable=protected-access
          (nodes["dW2_2"], "avg", 2, "group"),
          "rs_dW2_2",
      )
    graph.find_nodes(op="output")[0].replace_input_with(nodes["dW2_2"], rs)

    # Pass raw `fx.Graph` directly to `apply`.
    mark_activation_checkpoints.apply(graph)

    node_list = list(graph.nodes)
    barriers = _get_barriers(graph)
    self.assertLen(barriers, 2)
    # Non-straddling `fwd_getitem` is not reordered before `fwd_other`, while
    # straddling `inner_tup` and `leaf_tensor` are hoisted before `barriers[0]`.
    self.assertLess(node_list.index(fwd_other), node_list.index(fwd_getitem))
    self.assertLess(node_list.index(leaf_tensor), node_list.index(barriers[0]))

    b0_inputs = set(barriers[0].args[0])
    self.assertIn(wait_ag_w1_2, b0_inputs)
    self.assertIn(leaf_tensor, b0_inputs)
    self.assertTrue(
        b0_inputs.isdisjoint({leaf_none, scalar_seed, factory_ones, attr_const})
    )
    self.assertNotIn(rs, set(barriers[1].args[0]))

  def test_region_grouping_and_shared_weights(self):
    sample = torch.empty(4, 8)
    mm = torch.ops.aten.mm.default

    # 1. Chained un-saved remat nodes (`r0 -> r1`) merge into a single barrier.
    chained_g = fx.Graph()
    cx = chained_g.placeholder("cx")
    cx.meta["val"] = sample
    r0 = _add_remat_node(chained_g, torch.ops.aten.relu.default, (cx,), 0, "r0")
    r1 = _add_remat_node(chained_g, torch.ops.aten.relu.default, (r0,), 1, "r1")
    b1 = _add_bwd_node(chained_g, torch.ops.aten.neg.default, (r1,), "b1")
    chained_g.output((b1,))
    mark_activation_checkpoints.apply(chained_g)
    chained_barriers = _get_barriers(chained_g)
    self.assertLen(chained_barriers, 1)
    self.assertIs(_first_non_getitem_after(chained_barriers[0]), r0)

    # 2. Shared weights across regions (`ac_graph_id` 0 then 1) and a tuple
    #    remat op (`split.Tensor`) with a lazy `getitem` after the next region.
    graph = fx.Graph()
    x_1 = graph.placeholder("x_1")
    x_2 = graph.placeholder("x_2")
    w_shared = graph.placeholder("W_shared")
    dy = graph.placeholder("dY")
    for p in (x_1, x_2, w_shared, dy):
      p.meta["val"] = sample

    split_remat = _add_remat_node(
        graph, torch.ops.aten.split.Tensor, (x_1, 4), 0, "split_remat"
    )
    split_remat.meta["val"] = (sample, sample)
    remat_0 = graph.call_function(operator.getitem, (split_remat, 0))
    remat_0.meta.update(split_remat.meta)
    remat_0.meta["val"] = sample
    r_prev = _add_remat_node(graph, mm, (remat_0, w_shared), 0, "r_prev")
    dw_0 = _add_bwd_node(graph, mm, (r_prev, dy), "dW_0")
    dx_0 = _add_bwd_node(graph, mm, (dy, w_shared), "dX_0")

    r_next = _add_remat_node(graph, mm, (x_2, w_shared), 1, "r_next")
    remat_1_lazy = graph.call_function(operator.getitem, (split_remat, 1))
    remat_1_lazy.meta.update(split_remat.meta)
    remat_1_lazy.meta["val"] = sample
    dw_1 = _add_bwd_node(graph, mm, (r_next, dx_0), "dW_1")
    dx_1 = _add_bwd_node(graph, mm, (dx_0, w_shared), "dX_1")
    graph.output((dx_0, dw_0, dx_1, dw_1, remat_1_lazy))

    mark_activation_checkpoints.apply(graph)
    barriers = _get_barriers(graph)
    self.assertLen(barriers, 2)
    self.assertEqual(list(barriers[0].args[0]), [x_1, w_shared, dy])
    self.assertEqual(list(barriers[1].args[0]), [x_2, w_shared, dx_0])
    self.assertEqual(set(w_shared.users.keys()), set(barriers))

  def test_missing_val_metadata_raises(self):
    # Missing `'val'` metadata on a candidate barrier input raises.
    gm, nodes = _build_two_layer_graph(include_forward=False)
    del nodes["x_2"].meta["val"]
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Compiler pass unit test.
        ValueError, "Expected FX node 'x_2' to have 'val' metadata populated"
    ):
      mark_activation_checkpoints.apply(gm)

  def test_interleaved_regions_share_one_barrier(self):
    graph = fx.Graph()
    sample = torch.empty(4, 8)
    x_1 = graph.placeholder("x_1")
    x_2 = graph.placeholder("x_2")
    for p in (x_1, x_2):
      p.meta["val"] = sample
    relu = torch.ops.aten.relu.default
    a_2 = _add_remat_node(graph, relu, (x_2,), 1, "a_2")
    a_1 = _add_remat_node(graph, relu, (x_1,), 0, "a_1")
    b_2 = _add_remat_node(graph, relu, (a_2,), 1, "b_2")
    out = _add_bwd_node(graph, torch.ops.aten.add.Tensor, (a_1, b_2), "out")
    graph.output((out,))

    mark_activation_checkpoints.apply(graph)

    barriers = _get_barriers(graph)
    self.assertLen(barriers, 1)
    self.assertEqual(list(barriers[0].args[0]), [x_2, x_1])
    self.assertIs(_first_non_getitem_after(barriers[0]), a_2)

  def test_three_d_linear_checkpoint_saves_boundary_and_places_barriers(self):
    class FourLayerCheckpointedMLP(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.GELU(),
                torch.nn.Linear(16, 8),
            )
            for _ in range(4)
        ])

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
          # Flatten to 2D with `view` before `nn.Linear` and unflatten back to
          # 3D with `view` after `nn.Linear` so the checkpointed region's FX
          # graph (`view -> addmm -> gelu -> addmm -> view`) matches PyTorch
          # < 2.15's `_flatten_nd_linear` on all PyTorch versions.
          x = torch.utils.checkpoint.checkpoint(
              lambda inp, l=layer: l(inp.view(-1, inp.shape[-1])).view(
                  inp.shape
              ),
              x,
              use_reentrant=False,
          )
        return x

    fw_graphs: list[fx.GraphModule] = []
    bw_graphs: list[fx.GraphModule] = []

    def fw_compiler(gm: fx.GraphModule, example_inputs: Sequence[Any]):
      del example_inputs
      fw_graphs.append(copy.deepcopy(gm))
      return functorch_aot.make_boxed_func(gm.forward)

    # pylint: disable=protected-access
    tpu_compile_mod = torch_tpu._internal.compile
    tpu_backend_mod = tpu_compile_mod._backend

    def bw_compiler(gm: fx.GraphModule, example_inputs: Sequence[Any]):
      del example_inputs
      bw_copy = copy.deepcopy(gm)
      if test_utils.get_torch_version() < (2, 12):
        tpu_compile_mod.compiler._backfill_backward_graph_tags(bw_copy)
      mark_activation_checkpoints.apply(bw_copy)
      bw_graphs.append(bw_copy)
      return functorch_aot.make_boxed_func(gm.forward)

    backend = aot_autograd(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=tpu_backend_mod.min_cut_rematerialization_partition,
        decompositions=tpu_backend_mod._TPU_DECOMPOSITIONS,
        keep_inference_input_mutations=False,
    )
    # pylint: enable=protected-access
    for recompute_views in (False, True):
      with self.subTest(recompute_views=recompute_views):
        torch._dynamo.reset()
        fw_graphs.clear()
        bw_graphs.clear()
        with torch._functorch.config.patch(recompute_views=recompute_views):
          model = FourLayerCheckpointedMLP()
          x = torch.randn(2, 4, 8)
          compiled = torch.compile(model, backend=backend, fullgraph=True)
          out = compiled(x)
          out.sum().backward()

        self.assertLen(fw_graphs, 1)
        self.assertLen(bw_graphs, 1)
        bw_gm = bw_graphs[0]
        bwd_addmms = [
            n
            for n in bw_gm.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.addmm.default
        ]
        barriers = _get_barriers(bw_gm)
        self.assertLen(
            bwd_addmms,
            4,
            "Expected only 4 recomputed addmm ops in backward, got"
            f" {len(bwd_addmms)}:\n"
            f"{bw_gm.print_readable(print_output=False)}",
        )
        self.assertLen(
            barriers,
            4,
            "Expected 4 per-layer barriers in backward, got"
            f" {len(barriers)}:\n"
            f"{bw_gm.print_readable(print_output=False)}",
        )
        for barrier in barriers:
          for inp in barrier.args[0]:
            self.assertLessEqual(
                inp.meta["val"].ndim,
                2,
                f"Expected barrier input {inp.name!r} to be flattened to <= 2D,"
                f" got shape {tuple(inp.meta['val'].shape)}",
            )
        nodes_after_first = [
            (n.op, n.target, n.args) for n in bw_gm.graph.nodes
        ]
        mark_activation_checkpoints.apply(bw_gm)
        self.assertEqual(
            nodes_after_first,
            [(n.op, n.target, n.args) for n in bw_gm.graph.nodes],
        )

  def test_three_d_barrier_inputs_flattened_to_2d_and_idempotent(self):
    graph = fx.Graph()
    x_shared_3d = graph.placeholder("x_shared_3d")
    x_shared_3d.meta["val"] = torch.empty(2, 4, 8)
    x_4d = graph.placeholder("x_4d")
    x_4d.meta["val"] = torch.empty(2, 3, 4, 8)
    x_noncontig = graph.placeholder("x_noncontig")
    x_noncontig.meta["val"] = torch.empty(2, 8, 4).transpose(1, 2)
    x_empty = graph.placeholder("x_empty")
    x_empty.meta["val"] = torch.empty(2, 0, 8)
    dy_3d = graph.placeholder("dy_3d")
    dy_3d.meta["val"] = torch.empty(2, 4, 8)

    # Pre-region 2D view of `x_shared_3d` whose only consumer is a backward
    # `aten.view` inside Region 1 that reshapes back to `[2, 4, 8]`. Without
    # explicit `_BARRIER_VIEW_META_KEY` tagging on synthetic barrier views,
    # rewiring `v_pre` into Region 1's barrier would make `v_pre` and
    # `b1_user_unflat` match the local graph topology of synthetic flatten and
    # unflatten views, silently dropping `b1_user_unflat` from Region 0's
    # barrier.
    v_pre = _add_bwd_node(
        graph, torch.ops.aten.view.default, (x_shared_3d, [8, 8]), "v_pre"
    )
    v_pre.meta["val"] = torch.empty(8, 8)

    # Region 1 (ac_graph_id = 1): consumes contiguous 3D/4D, non-contiguous 3D,
    # and empty 3D inputs.
    r1_3d = _add_remat_node(
        graph, torch.ops.aten.gelu.default, (x_shared_3d,), 1, "r1_3d"
    )
    r1_3d.meta["val"] = torch.empty(2, 4, 8)
    r1_3d.meta["stack_trace"] = "r1_trace"
    r1_4d = _add_remat_node(
        graph, torch.ops.aten.relu.default, (x_4d,), 1, "r1_4d"
    )
    r1_4d.meta["val"] = torch.empty(2, 3, 4, 8)
    r1_empty = _add_remat_node(
        graph, torch.ops.aten.neg.default, (x_empty,), 1, "r1_empty"
    )
    r1_empty.meta["val"] = torch.empty(2, 0, 8)
    b1_user_unflat = _add_bwd_node(
        graph,
        torch.ops.aten.view.default,
        (v_pre, [2, 4, 8]),
        "b1_user_unflat",
    )
    b1_user_unflat.meta["val"] = torch.empty(2, 4, 8)
    b1_mul = _add_bwd_node(
        graph, torch.ops.aten.mul.Tensor, (r1_3d, dy_3d), "b1_mul"
    )
    b1_mul.meta["val"] = torch.empty(2, 4, 8)
    b1_unflat_term = _add_bwd_node(
        graph,
        torch.ops.aten.mul.Tensor,
        (r1_3d, b1_user_unflat),
        "b1_unflat_term",
    )
    b1_unflat_term.meta["val"] = torch.empty(2, 4, 8)
    b1_with_unflat = _add_bwd_node(
        graph,
        torch.ops.aten.add.Tensor,
        (b1_mul, b1_unflat_term),
        "b1_with_unflat",
    )
    b1_with_unflat.meta["val"] = torch.empty(2, 4, 8)
    b1_noncontig_term = _add_bwd_node(
        graph,
        torch.ops.aten.mul.Tensor,
        (r1_3d, x_noncontig),
        "b1_noncontig_term",
    )
    b1_noncontig_term.meta["val"] = torch.empty(2, 4, 8)
    b1_noncontig = _add_bwd_node(
        graph,
        torch.ops.aten.add.Tensor,
        (b1_with_unflat, b1_noncontig_term),
        "b1_noncontig",
    )
    b1_noncontig.meta["val"] = torch.empty(2, 4, 8)
    b1_4d = _add_bwd_node(
        graph, torch.ops.aten.mul.Tensor, (r1_4d, x_4d), "b1_4d"
    )
    b1_4d.meta["val"] = torch.empty(2, 3, 4, 8)
    b1_empty = _add_bwd_node(
        graph, torch.ops.aten.neg.default, (r1_empty,), "b1_empty"
    )
    b1_empty.meta["val"] = torch.empty(2, 0, 8)
    # Multi-hop backward consumer of `x_shared_3d` (analogous to `dW1`).
    x_2d_bwd = _add_bwd_node(
        graph, torch.ops.aten.view.default, (x_shared_3d, [8, 8]), "x_2d_bwd"
    )
    b1_2d_grad = _add_bwd_node(
        graph, torch.ops.aten.view.default, (b1_noncontig, [8, 8]), "b1_2d_grad"
    )
    dw_1 = _add_bwd_node(
        graph, torch.ops.aten.mm.default, (b1_2d_grad, x_2d_bwd), "dw_1"
    )

    # Region 0 (ac_graph_id = 0): consumes `x_shared_3d`, `b1_noncontig`, and
    # `b1_user_unflat`.
    r0_3d = _add_remat_node(
        graph, torch.ops.aten.gelu.default, (x_shared_3d,), 0, "r0_3d"
    )
    r0_3d.meta["val"] = torch.empty(2, 4, 8)
    dx_0_base = _add_bwd_node(
        graph, torch.ops.aten.mul.Tensor, (r0_3d, b1_noncontig), "dx_0_base"
    )
    dx_0_base.meta["val"] = torch.empty(2, 4, 8)
    dx_0_unflat_term = _add_bwd_node(
        graph,
        torch.ops.aten.mul.Tensor,
        (r0_3d, b1_user_unflat),
        "dx_0_unflat_term",
    )
    dx_0_unflat_term.meta["val"] = torch.empty(2, 4, 8)
    dx_0 = _add_bwd_node(
        graph,
        torch.ops.aten.add.Tensor,
        (dx_0_base, dx_0_unflat_term),
        "dx_0",
    )
    dx_0.meta["val"] = torch.empty(2, 4, 8)

    graph.output((dx_0, dw_1, b1_4d, b1_empty))
    gm = fx.GraphModule(torch.nn.Module(), graph)
    ref_gm = copy.deepcopy(gm)

    mark_activation_checkpoints.apply(gm)

    barriers = _get_barriers(gm)
    self.assertLen(barriers, 2)

    # Region 1 barrier: `x_shared_3d` ([8, 8]), `x_4d` ([24, 8]), and `dy_3d`
    # ([8, 8]) are flattened to 2D; `v_pre` is already 2D ([8, 8]);
    # `x_noncontig` stays 3D; `x_empty` (0-byte buffer) is excluded.
    b0_args = list(barriers[0].args[0])
    self.assertIn(v_pre, b0_args)
    self.assertIn(x_noncontig, b0_args)
    self.assertNotIn(x_empty, b0_args)
    for inp in b0_args:
      if inp is x_noncontig:
        self.assertEqual(inp.meta["val"].ndim, 3)
      else:
        self.assertEqual(inp.meta["val"].ndim, 2)
        self.assertEqual(inp.target, torch.ops.aten.view.default)
        if inp is not v_pre:
          self.assertEqual(inp.meta.get("stack_trace"), "r1_trace")

    # Region 0 barrier: `x_shared_3d`, `b1_noncontig`, and `b1_user_unflat` are
    # all flattened to 2D `[8, 8]` and gated by `barriers[1]`.
    b1_args = list(barriers[1].args[0])
    self.assertLen(b1_args, 3)
    for inp in b1_args:
      self.assertEqual(inp.meta["val"].shape, torch.Size([8, 8]))
      self.assertEqual(inp.target, torch.ops.aten.view.default)
    self.assertEqual(
        {inp.args[0] for inp in b1_args},
        {x_shared_3d, b1_noncontig, b1_user_unflat},
    )

    # `x_shared_3d` only feeds `v_pre` and the two pre-barrier 2D flatten views.
    self.assertLen(x_shared_3d.users, 3)
    for user in x_shared_3d.users:
      self.assertEqual(user.target, torch.ops.aten.view.default)

    # Verify numerical equivalence on TPU before vs. after the pass.
    device = torch.device("tpu")
    tpu_inputs = (
        torch.randn(2, 4, 8, device=device),
        torch.randn(2, 3, 4, 8, device=device),
        torch.randn(2, 8, 4, device=device).transpose(1, 2),
        torch.empty(2, 0, 8, device=device),
        torch.randn(2, 4, 8, device=device),
    )
    expected_outs = ref_gm(*tpu_inputs)
    actual_outs = gm(*tpu_inputs)
    for actual, expected in zip(actual_outs, expected_outs, strict=True):
      test_utils.assert_close(actual, expected)

    # Calling `apply` a second time is a strict no-op.
    nodes_after_first = [(n.op, n.target, n.args) for n in gm.graph.nodes]
    code_after_first = gm.code
    mark_activation_checkpoints.apply(gm)
    self.assertEqual(
        nodes_after_first, [(n.op, n.target, n.args) for n in gm.graph.nodes]
    )
    self.assertEqual(code_after_first, gm.code)

  def test_three_d_activation_checkpointing_on_tpu(self):
    cases: list[
        tuple[
            str,
            Callable[[], tuple[torch.nn.Module, torch.nn.Module]],
            bool,
            int,
            int,
        ]
    ] = [
        (
            "mlp_unfused",
            lambda: (
                _TwoLayerCheckpointedMLP(use_checkpoint=True),
                _TwoLayerCheckpointedMLP(use_checkpoint=False),
            ),
            False,
            2,
            10,
        ),
    ]
    if test_utils.get_torch_version() >= (2, 12):
      cases.extend([
          (
              "mlp_fused",
              lambda: (
                  _TwoLayerCheckpointedMLP(use_checkpoint=True),
                  _TwoLayerCheckpointedMLP(use_checkpoint=False),
              ),
              True,
              4,
              15,
          ),
          (
              "sac_fused",
              lambda: (
                  _TwoLayerSACModel(use_sac=True),
                  _TwoLayerSACModel(use_sac=False),
              ),
              True,
              4,
              12,
          ),
      ])

    for name, build_models, fuse_step, expected_erfs, expected_dots in cases:
      with self.subTest(name=name):
        torch._dynamo.reset()
        model, ref_model = build_models()
        debug = self._run_and_verify_tpu_model(
            model,
            ref_model,
            fuse_step=fuse_step,
            input_shape=(2, 4, 8),
        )
        mlir_text = (
            debug.stablehlo_forward_text[0]
            if fuse_step
            else debug.stablehlo_backward_text[0]
        )
        barrier_lines = [
            line
            for line in mlir_text.splitlines()
            if "stablehlo.optimization_barrier" in line
        ]
        self.assertLen(barrier_lines, 2)
        for line in barrier_lines:
          self.assertNotIn("tensor<2x4x8xf32>", line)
          self.assertIn("tensor<8x8xf32>", line)
        self._assert_hlo_cse_idempotent(
            expected_barriers=2,
            expected_erfs=expected_erfs,
            expected_dots=expected_dots,
            min_erfs_after_cse=expected_erfs,
        )

  def test_selective_activation_checkpointing_with_layernorm_on_tpu(self):
    debug = self._run_and_verify_tpu_model(
        _TwoLayerSACModel(use_sac=True),
        _TwoLayerSACModel(use_sac=False),
        fuse_step=False,
    )
    self.assertLen(debug.compiled_executables, 2)
    self.assertNotIn(
        "stablehlo.optimization_barrier", debug.stablehlo_forward_text[0]
    )
    bwd_mlir = debug.stablehlo_backward_text[0]
    self.assertEqual(bwd_mlir.count("stablehlo.optimization_barrier"), 2)
    self.assertEqual(bwd_mlir.count("chlo.erf"), 2)
    self._assert_hlo_cse_idempotent(
        expected_barriers=2, expected_erfs=2, expected_dots=8
    )

  def test_activation_checkpointing(self):
    debug = self._run_and_verify_tpu_model(
        _TwoLayerCheckpointedMLP(use_checkpoint=True),
        _TwoLayerCheckpointedMLP(use_checkpoint=False),
        fuse_step=False,
    )
    self.assertLen(debug.compiled_executables, 2)
    self.assertNotIn(
        "stablehlo.optimization_barrier", debug.stablehlo_forward_text[0]
    )
    bwd_mlir = debug.stablehlo_backward_text[0]
    self.assertEqual(bwd_mlir.count("stablehlo.optimization_barrier"), 2)
    self.assertEqual(bwd_mlir.count("chlo.erf"), 2)
    # 8 backward aten.mm ops lower to 8 stablehlo.dot_general ops in MLIR; XLA
    # HLO additionally lowers the 2 recomputed aten.addmm ops to 2 dot( ops
    # (8 + 2 = 10 total dot( ops in HLO).
    self.assertEqual(bwd_mlir.count("stablehlo.dot_general"), 8)
    self._assert_hlo_cse_idempotent(
        expected_barriers=2, expected_erfs=2, expected_dots=10
    )

  def test_activation_checkpointing_fused_step(self):
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not
    # automatically detect traced `loss.backward()` nodes
    # (`autograd_backward=True`, pytorch/pytorch#179105) or tag duplicated
    # remat nodes (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")

    # 1. Negative control: without barriers, MLIR CSE deduplicates the 2
    #    backward recomputed `chlo.erf` ops against the 2 forward `chlo.erf` ops
    #    (collapsing 4 -> 2).
    with mock.patch.object(
        mark_activation_checkpoints, "apply", side_effect=lambda gm: None
    ):
      no_barrier_debug = self._run_and_verify_tpu_model(
          _TwoLayerCheckpointedMLP(use_checkpoint=True),
          _TwoLayerCheckpointedMLP(use_checkpoint=False),
          fuse_step=True,
      )
    no_barrier_mlir = no_barrier_debug.stablehlo_forward_text[0]
    self.assertNotIn("stablehlo.optimization_barrier", no_barrier_mlir)
    self.assertEqual(no_barrier_mlir.count("chlo.erf"), 2)

    # 2. With barriers enabled, all 4 `chlo.erf` ops and 15 HLO `dot(` ops
    #    (4 forward addmm + 3 recomputed addmm + 8 backward mm) survive both
    #    MLIR CSE and XLA `HloCSE`.
    torch._dynamo.reset()
    debug = self._run_and_verify_tpu_model(
        _TwoLayerCheckpointedMLP(use_checkpoint=True),
        _TwoLayerCheckpointedMLP(use_checkpoint=False),
        fuse_step=True,
    )
    self.assertLen(debug.compiled_executables, 1)
    fused_mlir = debug.stablehlo_forward_text[0]
    self.assertEqual(fused_mlir.count("stablehlo.optimization_barrier"), 2)
    self.assertEqual(fused_mlir.count("chlo.erf"), 4)
    self.assertEqual(fused_mlir.count("stablehlo.dot_general"), 8)
    self._assert_hlo_cse_idempotent(
        expected_barriers=2, expected_erfs=4, expected_dots=15
    )

  def test_selective_activation_checkpointing_with_layernorm_fused_step(self):
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not
    # automatically detect traced `loss.backward()` nodes
    # (`autograd_backward=True`, pytorch/pytorch#179105) or tag duplicated
    # remat nodes (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")

    debug = self._run_and_verify_tpu_model(
        _TwoLayerSACModel(use_sac=True),
        _TwoLayerSACModel(use_sac=False),
        fuse_step=True,
    )
    self.assertLen(debug.compiled_executables, 1)
    fused_mlir = debug.stablehlo_forward_text[0]
    self.assertEqual(fused_mlir.count("stablehlo.optimization_barrier"), 2)
    self.assertEqual(fused_mlir.count("chlo.erf"), 4)
    # Under `_sac_policy_fn`, the 4 forward addmm ops are saved (`MUST_SAVE`)
    # and not recomputed in backward, yielding 4 forward addmm + 8 backward mm =
    # 12 HLO `dot(` ops.
    self._assert_hlo_cse_idempotent(
        expected_barriers=2, expected_erfs=4, expected_dots=12
    )

  def test_activation_checkpointing_shared_weights_fused_step(self):
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not
    # automatically detect traced `loss.backward()` nodes
    # (`autograd_backward=True`, pytorch/pytorch#179105) or tag duplicated
    # remat nodes (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")

    debug = self._run_and_verify_tpu_model(
        _SharedWeightCheckpointedMLP(use_checkpoint=True),
        _SharedWeightCheckpointedMLP(use_checkpoint=False),
        fuse_step=True,
    )
    self.assertLen(debug.compiled_executables, 1)
    fused_mlir = debug.stablehlo_forward_text[0]
    self.assertEqual(fused_mlir.count("stablehlo.optimization_barrier"), 2)
    self.assertEqual(fused_mlir.count("chlo.erf"), 4)
    self.assertEqual(fused_mlir.count("stablehlo.dot_general"), 8)
    self._assert_hlo_cse_idempotent(
        expected_barriers=2, expected_erfs=4, expected_dots=15
    )


if __name__ == "__main__":
  absltest.main()
