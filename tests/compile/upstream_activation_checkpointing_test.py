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

"""Ported upstream PyTorch activation checkpointing tests for TorchTPU.

Provenance: `test/dynamo/test_activation_checkpointing.py`

NOTE: This file is temporary. We plan to retire it in favor of running the
upstream PyTorch activation checkpointing test suite directly on TPU in the
future.
"""

from collections.abc import Callable
import contextlib
import copy
import dataclasses
import functools
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch._dynamo.exc
import torch._functorch.config
import torch._functorch.partitioners
from torch import fx
from torch import nn
import torch.nn.functional as F
from torch.utils import _pytree as pytree
from torch.utils import checkpoint
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.compile.fx_passes import mark_activation_checkpoints
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


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


# ==============================================================================
# Ported Upstream PyTorch Activation Checkpointing Tests
# Provenance: test/dynamo/test_activation_checkpointing.py
# ==============================================================================

_default_partition = torch._functorch.partitioners.default_partition  # pylint: disable=protected-access
_min_cut_partition = (
    torch._functorch.partitioners.min_cut_rematerialization_partition  # pylint: disable=protected-access
)
_tag_activation_checkpoint = (
    torch._higher_order_ops.wrap.tag_activation_checkpoint  # pylint: disable=protected-access
)

_PARTITION_FN_PARAMS = (
    ("min_cut", _min_cut_partition),
    ("default", _default_partition),
)


def _grad(*args, **kwargs):
  # Provenance: test_activation_checkpointing.py::_grad
  return torch.autograd.grad(*args, **kwargs)


def _count_ops(
    gm: fx.GraphModule,
    args: Any,
    freq: int | None = None,
    freq_ge: int | None = None,
    op: Any = None,
    freqs: list[int] | None = None,
    freqs_ge: list[int] | None = None,
    ops: list[Any] | None = None,
) -> fx.GraphModule:
  # Provenance: test_activation_checkpointing.py::count_ops
  del args

  def match_rng_op(node: fx.Node, target_op: Any) -> bool:
    if isinstance(node.target, torch._ops.HigherOrderOperator):  # pylint: disable=protected-access
      if node.name == "run_and_save_rng_state":
        return node.args[0] == target_op
      elif node.name == "run_with_rng_state":
        return node.args[1] == target_op
      elif node.name == "graphsafe_run_with_rng_state":
        return node.args[0] == target_op
    return False

  if op is not None:
    if isinstance(op, list):
      raise AssertionError("Expected op to not be a list")
    ops = [op]
  if freq is not None:
    freqs = [freq]
  if freq_ge is not None:
    freqs_ge = [freq_ge]
  assert ops is not None
  if freqs:
    for target_op, expected_freq in zip(ops, freqs):
      actual_count = 0
      for node in gm.graph.nodes:
        if match_rng_op(node, target_op) or node.target == target_op:
          actual_count += 1
      if actual_count != expected_freq:
        raise AssertionError(
            f"In graph {gm}, expected {target_op} to have occurred"
            f" {expected_freq} times in the graph, but got {actual_count}."
        )
  else:
    if freqs_ge is None:
      raise AssertionError("Expected freqs_ge to not be None")
    for target_op, expected_freq_ge in zip(ops, freqs_ge):
      actual_count = 0
      for node in gm.graph.nodes:
        if match_rng_op(node, target_op) or node.target == target_op:
          actual_count += 1
      if actual_count < expected_freq_ge:
        raise AssertionError(
            f"In graph {gm}, expected {target_op} to have occurred at least"
            f" {expected_freq_ge} times in the graph, but got {actual_count}."
        )
  return gm


def _find_first_node(gm: fx.GraphModule, func: Any) -> fx.Node | None:
  # Provenance: test_activation_checkpointing.py::find_first_node
  for node in gm.graph.nodes:
    if node.target is func:
      return node
  return None


def _op_count(gm: fx.GraphModule) -> int:
  # Provenance: test_activation_checkpointing.py::op_count
  result = 0
  for node in gm.graph.nodes:
    if "call" in node.op:
      result += 1
  return result


def _get_custom_policy(no_recompute_list=None, must_recompute_list=None):
  # Provenance: test_activation_checkpointing.py::_get_custom_policy
  def _custom_policy(ctx, func, *args, **kwargs):
    del ctx, args, kwargs
    if no_recompute_list is not None and func in no_recompute_list:
      return checkpoint.CheckpointPolicy.MUST_SAVE
    if must_recompute_list is not None and func in must_recompute_list:
      return checkpoint.CheckpointPolicy.MUST_RECOMPUTE
    else:
      return checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

  return _custom_policy


class _TpuRecordingBackend:
  """Wraps `TpuBackend` to run FX graph checks and record graphs while compiling for TPU."""

  def __init__(
      self,
      fw_compiler: Callable[..., Any] | None = None,
      bw_compiler: Callable[..., Any] | None = None,
      partition_fn: Callable[..., Any] = _min_cut_partition,
      decompositions: Any = None,
  ):
    self._fw_compiler = fw_compiler
    self._bw_compiler = bw_compiler
    self._partition_fn = partition_fn
    self._decompositions = decompositions
    self.graphs: list[fx.GraphModule] = []
    self.fw_graphs: list[fx.GraphModule] = []
    self.bw_graphs: list[fx.GraphModule] = []
    self.frame_count = 0

  def __call__(
      self,
      gm: fx.GraphModule,
      example_inputs: list[torch.Tensor],
      **kwargs: Any,
  ) -> Callable[..., Any]:
    self.frame_count += 1
    self.graphs.append(copy.deepcopy(gm))
    backend_mod = torch_tpu_compile._backend  # pylint: disable=protected-access
    tpu_backend = backend_mod.TpuBackend(enable_serialization=False)
    orig_compile_gm = tpu_backend._compile_graph_module  # pylint: disable=protected-access

    def wrapped_compile_gm(
        compiler_instance,
        is_fwd,
        custom_module_name,
        debug,
        compiled_artifacts,
        input_layouts,
        graph_module,
        inputs,
    ):
      if is_fwd:
        self.fw_graphs.append(copy.deepcopy(graph_module))
        if self._fw_compiler is not None:
          self._fw_compiler(graph_module, inputs)
      else:
        self.bw_graphs.append(copy.deepcopy(graph_module))
        if self._bw_compiler is not None:
          self._bw_compiler(graph_module, inputs)
      return orig_compile_gm(
          compiler_instance,
          is_fwd,
          custom_module_name,
          debug,
          compiled_artifacts,
          input_layouts,
          graph_module,
          inputs,
      )

    decomps = (
        self._decompositions()
        if callable(self._decompositions)
        else self._decompositions
    )
    with contextlib.ExitStack() as stack:
      stack.enter_context(
          mock.patch.object(
              tpu_backend,
              "_compile_graph_module",
              side_effect=wrapped_compile_gm,
          )
      )
      stack.enter_context(
          mock.patch.object(
              backend_mod,
              "min_cut_rematerialization_partition",
              self._partition_fn,
          )
      )
      if decomps is not None:
        stack.enter_context(
            mock.patch.object(backend_mod, "_TPU_DECOMPOSITIONS", decomps)
        )
      return tpu_backend(gm, example_inputs, **kwargs)


class ActivationCheckpointingViaTagsTests(seed_test_utils.RepeatableTest):
  """Provenance: `test/dynamo/test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests`."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _validate(
      self,
      fn: Callable[..., Any],
      backend: Any,
      *args: torch.Tensor,
      skip_check: bool = False,
      fullgraph: bool = True,
  ) -> None:
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests._validate
    tpu_args = [
        arg.to(self.device).detach().clone().requires_grad_(arg.requires_grad)
        for arg in args
    ]
    cloned_args = [
        arg.detach().clone().requires_grad_(arg.requires_grad)
        for arg in tpu_args
    ]
    if isinstance(fn, nn.Module):
      fn = fn.to(self.device)
    cloned_fn = copy.deepcopy(fn)

    torch.manual_seed(0)
    expected = fn(*tpu_args)
    expected.sum().backward()

    torch.manual_seed(0)
    compiled_fn = torch.compile(cloned_fn, fullgraph=fullgraph, backend=backend)
    result = compiled_fn(*cloned_args)
    result.sum().backward()

    if not skip_check:
      test_utils.assert_close(result, expected)
      for arg, cloned_arg in zip(tpu_args, cloned_args):
        test_utils.assert_close(arg.grad, cloned_arg.grad)

  def _compare_orig_and_checkpointed_fns(
      self,
      orig_fn: Callable[..., Any],
      checkpointed_fn: Callable[..., Any],
      *args: torch.Tensor,
      fullgraph: bool = True,
  ) -> None:
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests._compare_orig_and_checkpointed_fns
    def clone_args(inp_args):
      return [
          arg.to(self.device).detach().clone().requires_grad_(arg.requires_grad)
          for arg in inp_args
      ]

    cloned_args_orig = clone_args(args)
    torch.manual_seed(0)
    compiled_orig_fn = torch.compile(
        orig_fn,
        backend="tpu",
        fullgraph=fullgraph,
        options={"serializable": False},
    )
    result_orig = compiled_orig_fn(*cloned_args_orig)
    result_orig.sum().backward()

    cloned_args_ckpt = clone_args(args)
    torch.manual_seed(0)
    compiled_ckpt_fn = torch.compile(
        copy.deepcopy(checkpointed_fn),
        backend="tpu",
        fullgraph=fullgraph,
        options={"serializable": False},
    )
    result_ckpt = compiled_ckpt_fn(*cloned_args_ckpt)
    result_ckpt.sum().backward()

    test_utils.assert_close(result_orig, result_ckpt)
    for arg_orig, arg_ckpt in zip(cloned_args_orig, cloned_args_ckpt):
      test_utils.assert_close(arg_orig.grad, arg_ckpt.grad)

  def _get_sac_annotations(
      self, checkpointed_fn: Callable[..., Any], policy_fn: Callable[..., Any]
  ) -> str:
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests._get_sac_annotations
    annotations: list[str] = []

    def capture_partition(joint_gm, joint_args, **kwargs):
      for node in joint_gm.graph.nodes:
        if node.op == "call_function":
          recompute = node.meta.get("recompute", None)
          if recompute is not None:
            annotations.append(
                f"{node.name}: {node.target} -> {recompute.name}"
            )
      return _min_cut_partition(joint_gm, joint_args, **kwargs)

    backend = _TpuRecordingBackend(partition_fn=capture_partition)

    def fn(x):
      return checkpoint.checkpoint(
          checkpointed_fn,
          x,
          use_reentrant=False,
          context_fn=functools.partial(
              checkpoint.create_selective_checkpoint_contexts, policy_fn
          ),
      )

    x = torch.randn(4, device=self.device, requires_grad=True)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=backend)
    out = compiled(x)
    out.sum().backward()
    return "\n".join(annotations)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_function(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_function
    def gn(x, y):
      return torch.sigmoid(torch.matmul(x, y))

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn, torch.sin(x), y, use_reentrant=True
      )

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=3, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_function_via_global_checkpoint(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_function_via_global_checkpoint
    def gn(x, y):
      return torch.sigmoid(torch.matmul(x, y))

    def fn(x, y):
      return checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=True)

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=3, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_function_with_kwargs(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_function_with_kwargs
    def gn(x, y):
      return torch.sigmoid(torch.matmul(x, y))

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn, torch.sin(x), y, use_reentrant=False
      )

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=3, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)

  def test_ac_tags_through_custom_autograd_function(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_ac_tags_through_custom_autograd_function
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest(
          "Propagating AC tags through custom autograd.Function requires"
          " PyTorch >= 2.12"
      )

    class MyMM(torch.autograd.Function):

      @staticmethod
      def forward(x, w):
        return x @ w

      @staticmethod
      def setup_context(ctx, inputs, output):
        del output
        x, w = inputs
        ctx.save_for_backward(x, w)

      @staticmethod
      def backward(ctx, grad):
        x, w = ctx.saved_tensors
        return grad @ w.t(), x.t() @ grad

    def gn(x, w):
      return MyMM.apply(x, w)

    def fn(x, w):
      return torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    w = torch.randn(4, 4, device=self.device, requires_grad=True)

    def partition_fn(joint_gm, *args, **kwargs):
      fwd_mm_nodes = [
          node
          for node in joint_gm.graph.nodes
          if node.op == "call_function"
          and node.target == torch.ops.aten.mm.default
          and node.meta.get("partitioner_tag") == "is_forward"
      ]
      self.assertNotEmpty(fwd_mm_nodes)
      for node in fwd_mm_nodes:
        self.assertIn("recompute", node.meta)
        self.assertIn("ac_graph_id", node.meta)
      return _min_cut_partition(joint_gm, *args, **kwargs)

    backend = _TpuRecordingBackend(partition_fn=partition_fn)
    out = torch.compile(fn, backend=backend)(x, w)
    out.sum().backward()

  def test_sac_tags_through_custom_autograd_function(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_sac_tags_through_custom_autograd_function
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest(
          "Propagating SAC tags through custom autograd.Function requires"
          " PyTorch >= 2.12"
      )

    class MyMM(torch.autograd.Function):

      @staticmethod
      def forward(x, w):
        return x @ w

      @staticmethod
      def setup_context(ctx, inputs, output):
        del output
        x, w = inputs
        ctx.save_for_backward(x, w)

      @staticmethod
      def backward(ctx, grad):
        x, w = ctx.saved_tensors
        return grad @ w.t(), x.t() @ grad

    def gn(x, w):
      return MyMM.apply(x, w)

    context_fn = functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts,
        lambda ctx, op, *args, **kwargs: (
            torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
        ),
    )

    def fn(x, w):
      return torch.utils.checkpoint.checkpoint(
          gn, x, w, use_reentrant=False, context_fn=context_fn
      )

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    w = torch.randn(4, 4, device=self.device, requires_grad=True)

    def partition_fn(joint_gm, *args, **kwargs):
      fwd_mm_nodes = [
          node
          for node in joint_gm.graph.nodes
          if node.op == "call_function"
          and node.target == torch.ops.aten.mm.default
          and node.meta.get("partitioner_tag") == "is_forward"
      ]
      self.assertNotEmpty(fwd_mm_nodes)
      for node in fwd_mm_nodes:
        self.assertIn("recompute", node.meta)
        self.assertIn("ac_graph_id", node.meta)
      return _min_cut_partition(joint_gm, *args, **kwargs)

    backend = _TpuRecordingBackend(partition_fn=partition_fn)
    out = torch.compile(fn, backend=backend)(x, w)
    out.sum().backward()

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_sequential_layers(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_sequential_layers
    def gn(x):
      x = x.cos()
      for _ in range(3):
        x = torch.mm(x, x)
      x = x.cos()
      return x

    def fn(x):
      x = torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=False)
      x = torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=False)
      return x

    x = torch.randn(4, 4, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=6, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops,
        freqs=[2, 18],
        ops=[torch.ops.aten.cos.default, torch.ops.aten.mm.default],
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_multiple_checkpoints(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_multiple_checkpoints
    def gn(x, y):
      return torch.sigmoid(torch.matmul(x, y))

    def fn(x, y):
      x = torch.sin(x)
      z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
      x = torch.sin(z)
      z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
      return z

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=2, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=6, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_module(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_module
    class MockModule(torch.nn.Module):

      def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

      def forward(self, x):
        return torch.sigmoid(self.linear(x))

    mod = MockModule().to(self.device)

    def fn(x):
      return torch.utils.checkpoint.checkpoint(
          mod, torch.sin(x), use_reentrant=True
      )

    x = torch.randn(10, 10, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.sigmoid.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.sigmoid.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_tags_decomps(self, partition_fn):
    # Provenance (modified): test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_decomps
    # Modified from upstream: uses `torch._decomp.get_decompositions` for `gelu`
    # instead of `torch._inductor.compile_fx.select_decomp_table()`.
    class MockModule(torch.nn.Module):

      def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

      def forward(self, x):
        return torch.nn.functional.gelu(self.linear(x))

    mod = MockModule().to(self.device)

    def fn(x):
      return torch.utils.checkpoint.checkpoint(
          mod, torch.sin(x), use_reentrant=True
      )

    x = torch.randn(10, 10, device=self.device, requires_grad=True)

    fw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.erf.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=1, op=torch.ops.aten.erf.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
        decompositions=lambda: torch._decomp.get_decompositions([  # pylint: disable=protected-access
            torch.ops.aten.gelu.default,
            torch.ops.aten.gelu_backward.default,
        ]),
    )
    self._validate(fn, backend, x)

  def test_tags_recomputed_rand(self):
    # Provenance (modified): test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_recomputed_rand
    # Modified from upstream: omits `@torch._inductor.config.patch(fallback_random=True)`,
    # sets `skip_check=True` (since `graphsafe_rng_functionalization` changes RNG
    # numerics relative to eager), and verifies optimization barrier insertion.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `graphsafe_rng_functionalization` did not preserve
    # `recompute`/`ac_graph_id` tags on replaced RNG HOP nodes.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest(
          "Preserving AC tags across graphsafe_rng_functionalization requires"
          " PyTorch >= 2.12"
      )

    def gn(x, y):
      return torch.sigmoid(torch.rand_like(x) * y) * x

    def fn(x, y):
      x = torch.sin(x)
      x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
      x = torch.sin(x)
      z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
      return z

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)
    backend = _TpuRecordingBackend()
    self._validate(fn, backend, x, y, skip_check=True)
    mark_activation_checkpoints.apply(backend.bw_graphs[0])
    self.assertLen(_get_barriers(backend.bw_graphs[0].graph), 2)

  def test_tags_rand(self):
    # Provenance (modified): test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_rand
    # Modified from upstream: omits `@torch._inductor.config.patch(fallback_random=True)`.
    def gn(x, y):
      x = torch.mm(x, y)
      x = torch.mm(x, y)
      return x

    def fn(x, y):
      x = torch.sin(x)
      x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
      x = torch.sin(x)
      return x

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)
    self._validate(fn, _TpuRecordingBackend(), x, y)

  def test_tags_dropout(self):
    # Provenance (modified): test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_tags_dropout
    # Modified from upstream: omits `@torch._inductor.config.patch(fallback_random=True)`
    # and verifies optimization barrier insertion on the recomputed dropout op.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `graphsafe_rng_functionalization` did not preserve
    # `recompute`/`ac_graph_id` tags on replaced RNG HOP nodes.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest(
          "Preserving AC tags across graphsafe_rng_functionalization requires"
          " PyTorch >= 2.12"
      )

    class MockModule(torch.nn.Module):

      def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.dropout = torch.nn.Dropout(0.2)

      def forward(self, x):
        return self.dropout(self.linear(x))

    mod = MockModule().to(self.device)

    def fn(x):
      return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

    x = torch.randn(10, 10, device=self.device, requires_grad=True)
    backend = _TpuRecordingBackend()
    self._validate(fn, backend, x, skip_check=True)
    mark_activation_checkpoints.apply(backend.bw_graphs[0])
    self.assertLen(_get_barriers(backend.bw_graphs[0].graph), 1)

  def test_kwargs(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_kwargs
    def gn(x, y, z=None):
      a = torch.matmul(x, y)
      if z is not None:
        return torch.matmul(a, z)
      return a

    def fn(x, y, z):
      return torch.cos(
          checkpoint.checkpoint(gn, x, y, use_reentrant=False, z=z)
      )

    x = torch.randn(4, 4, requires_grad=True, device=self.device)
    y = torch.randn(4, 4, requires_grad=True, device=self.device)
    z = torch.randn(4, 4, requires_grad=True, device=self.device)
    args = (x, y, z)

    cnt = _TpuRecordingBackend()
    expected = fn(*args)
    result = torch.compile(fn, backend=cnt)(*args)

    test_utils.assert_close(result, expected)
    self.assertEqual(cnt.frame_count, 1)
    self.assertLen(cnt.graphs, 1)

    wrap_node = _find_first_node(cnt.graphs[0], _tag_activation_checkpoint)
    self.assertIsNotNone(wrap_node)
    self.assertLen(wrap_node.args, 4)

    body_function = getattr(cnt.graphs[0], wrap_node.args[0].name)
    self.assertEqual(_op_count(body_function), 2)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_must_recompute(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_must_recompute
    def context_fn_must_recompute_mm():
      return checkpoint.create_selective_checkpoint_contexts(
          _get_custom_policy(must_recompute_list=[torch.ops.aten.mm.default])
      )

    def context_fn_no_recompute_mm():
      return checkpoint.create_selective_checkpoint_contexts(
          _get_custom_policy(no_recompute_list=[torch.ops.aten.mm.default])
      )

    def _test(context_fn, bw_compiler):
      def gn(x):
        return torch.cos(torch.sin(torch.matmul(x, x) @ x))

      def fn(x):
        return torch.utils.checkpoint.checkpoint(
            gn, x, use_reentrant=False, context_fn=context_fn
        )

      x = torch.randn(4, 4, requires_grad=True, device=self.device)
      fw_compiler = functools.partial(
          _count_ops, freq=2, op=torch.ops.aten.mm.default
      )
      backend = _TpuRecordingBackend(
          fw_compiler=fw_compiler,
          bw_compiler=bw_compiler,
          partition_fn=partition_fn,
      )
      self._validate(fn, backend, x)

    _test(
        context_fn=context_fn_must_recompute_mm,
        bw_compiler=functools.partial(
            _count_ops, freq=6, op=torch.ops.aten.mm.default
        ),
    )
    torch._dynamo.reset()
    _test(
        context_fn=context_fn_no_recompute_mm,
        bw_compiler=functools.partial(
            _count_ops, freq=4, op=torch.ops.aten.mm.default
        ),
    )

  def test_sac_with_partial_context_fn(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_sac_with_partial_context_fn
    class CustomPolicy:

      def __call__(self, ctx, out, func, *args, **kwargs):
        del ctx, out, func, args, kwargs
        return checkpoint.CheckpointPolicy.MUST_SAVE

    def f(x, y):
      return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

    context_fn1 = functools.partial(
        checkpoint.create_selective_checkpoint_contexts, CustomPolicy()
    )

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          f, x, y, use_reentrant=False, context_fn=context_fn1
      )

    opt_fn = torch.compile(fn, backend=_TpuRecordingBackend(), fullgraph=True)
    a = torch.randn(4, 4, requires_grad=True, device=self.device)
    b = torch.randn(4, 4, requires_grad=True, device=self.device)

    expected = fn(a, b)
    result = opt_fn(a, b)
    test_utils.assert_close(result, expected)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_must_not_recompute_gemm(
      self, partition_fn
  ):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_must_not_recompute_gemm
    def selective_checkpointing_context_fn():
      return checkpoint.create_selective_checkpoint_contexts(
          _get_custom_policy(no_recompute_list=[torch.ops.aten.mm.default])
      )

    def gn(x, y):
      return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn,
          x,
          y,
          use_reentrant=False,
          context_fn=selective_checkpointing_context_fn,
      )

    x = torch.randn(4, 4, requires_grad=True, device=self.device)
    y = torch.randn(4, 4, requires_grad=True, device=self.device)

    fw_compiler = functools.partial(
        _count_ops, freq=2, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=4, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)
    self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_custom_rule(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_custom_rule
    def _make_custom_policy(meta):
      no_recompute_list = [torch.ops.aten.mm.default]

      def _custom_policy(mode, func, *args, **kwargs):
        del args, kwargs
        mm_count_key = f"{mode}_mm_count"
        if mm_count_key not in meta:
          meta[mm_count_key] = 0
        if func == torch.ops.aten.mm.default:
          meta[mm_count_key] += 1
        return func in no_recompute_list and not (
            func == torch.ops.aten.mm.default and meta[mm_count_key] == 2
        )

      return _custom_policy

    def selective_checkpointing_context_fn():
      meta = {}
      return checkpoint.create_selective_checkpoint_contexts(
          _make_custom_policy(meta)
      )

    def gn(x, y):
      return torch.sigmoid(
          torch.sigmoid(torch.matmul(torch.matmul(x, y) * y, y) * y)
      )

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn,
          x,
          y,
          use_reentrant=False,
          context_fn=selective_checkpointing_context_fn,
      )

    x = torch.randn(4, 4, requires_grad=True, device=self.device)
    y = torch.randn(4, 4, requires_grad=True, device=self.device)

    fw_compiler = functools.partial(
        _count_ops, freq=2, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq_ge=4, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)
    self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_partial_ctx_fn(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_partial_ctx_fn
    def selective_checkpointing_context_fn(no_recompute_list):
      return checkpoint.create_selective_checkpoint_contexts(
          _get_custom_policy(no_recompute_list=no_recompute_list)
      )

    def gn(x, y):
      return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn,
          x,
          y,
          use_reentrant=False,
          context_fn=functools.partial(
              selective_checkpointing_context_fn, [torch.ops.aten.mm.default]
          ),
      )

    x = torch.randn(4, 4, requires_grad=True, device=self.device)
    y = torch.randn(4, 4, requires_grad=True, device=self.device)

    fw_compiler = functools.partial(
        _count_ops, freq=2, op=torch.ops.aten.mm.default
    )
    bw_compiler = functools.partial(
        _count_ops, freq=4, op=torch.ops.aten.mm.default
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)
    self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_outplace_op(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_outplace_op
    def selective_checkpointing_context_fn():
      no_recompute_list = [
          torch.ops.aten.mm.default,
          torch.ops.aten.sigmoid.default,
      ]
      return checkpoint.create_selective_checkpoint_contexts(
          _get_custom_policy(no_recompute_list=no_recompute_list),
      )

    def gn(x, y):
      return torch.sigmoid(
          torch.selu(torch.matmul(torch.matmul(x, y), y))
      ).relu()

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn,
          x,
          y,
          use_reentrant=False,
          context_fn=selective_checkpointing_context_fn,
      )

    x = torch.randn(4, 4, requires_grad=True, device=self.device)
    y = torch.randn(4, 4, requires_grad=True, device=self.device)

    fw_compiler = functools.partial(
        _count_ops,
        freqs=[2, 1],
        ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
    )
    bw_compiler = functools.partial(
        _count_ops,
        freqs=[4, 0],
        ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)
    self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_list_ops(self, partition_fn):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_list_ops
    def selective_checkpointing_context_fn():
      return checkpoint.create_selective_checkpoint_contexts(
          _get_custom_policy(no_recompute_list=[])
      )

    def gn(x, y):
      return torch.cat([x, y]).sin()

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          gn,
          x,
          y,
          use_reentrant=False,
          context_fn=selective_checkpointing_context_fn,
      )

    x = torch.randn(4, 4, requires_grad=True, device=self.device)
    y = torch.randn(4, 4, requires_grad=True, device=self.device)

    fw_compiler = functools.partial(
        _count_ops, freqs=[1], ops=[torch.ops.aten.cat.default]
    )
    bw_compiler = functools.partial(
        _count_ops, freqs=[1], ops=[torch.ops.aten.cat.default]
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )
    self._validate(fn, backend, x, y)
    self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_random_op(self, partition_fn):
    # Provenance (modified): test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_random_op
    # Modified from upstream: omits `@torch._inductor.config.patch(fallback_random=True)`
    # and sets `skip_check=True` (since `graphsafe_rng_functionalization` changes RNG
    # numerics relative to eager/uncheckpointed).
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `graphsafe_rng_functionalization` did not preserve
    # `recompute`/`ac_graph_id` tags on replaced RNG HOP nodes.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest(
          "Preserving AC tags across graphsafe_rng_functionalization requires"
          " PyTorch >= 2.12"
      )
    for preserve_rng_state in [True, False]:
      torch._dynamo.reset()

      def selective_checkpointing_context_fn():
        return checkpoint.create_selective_checkpoint_contexts(
            _get_custom_policy(
                no_recompute_list=[torch.ops.aten.sigmoid.default]
            )
        )

      def gn(x):
        return torch.sigmoid(torch.dropout(torch.sigmoid(x), p=0.5, train=True))

      def fn(x, preserve=preserve_rng_state):
        return torch.utils.checkpoint.checkpoint(
            gn,
            x,
            use_reentrant=False,
            preserve_rng_state=preserve,
            context_fn=selective_checkpointing_context_fn,
        )

      x = torch.randn(4, 4, requires_grad=True, device=self.device)

      fw_compiler = functools.partial(
          _count_ops,
          freqs=[2, 1],
          ops=[
              torch.ops.aten.sigmoid.default,
              torch.ops.aten.native_dropout.default,
          ],
      )
      bw_compiler = functools.partial(
          _count_ops,
          freqs=[0, 1],
          ops=[
              torch.ops.aten.sigmoid.default,
              torch.ops.aten.native_dropout.default,
          ],
      )
      backend = _TpuRecordingBackend(
          fw_compiler=fw_compiler,
          bw_compiler=bw_compiler,
          partition_fn=partition_fn,
      )
      self._validate(fn, backend, x, skip_check=True)

  @parameterized.named_parameters(*_PARTITION_FN_PARAMS)
  def test_compile_selective_checkpoint_parametrization(self, partition_fn):
    # Provenance (modified): test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_compile_selective_checkpoint_parametrization
    # Modified from upstream: runs `MLPModule` directly without CUDA FSDP wrapping.
    def sac_policy():
      def _custom_policy(ctx, func, *args, **kwargs):
        del ctx, args, kwargs
        to_recompute = func in {
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.sigmoid.default,
        }
        return (
            checkpoint.CheckpointPolicy.MUST_RECOMPUTE
            if to_recompute
            else checkpoint.CheckpointPolicy.MUST_SAVE
        )

      return checkpoint.create_selective_checkpoint_contexts(_custom_policy)

    class Parametrization(torch.nn.Module):

      def parametrization(self, x):
        return torch.sigmoid(torch.mul(x, x))

      def forward(self, x):
        return checkpoint.checkpoint(
            self.parametrization, x, use_reentrant=False, context_fn=sac_policy
        )

    def apply_parametrization(model):
      for mod in list(model.modules()):
        params_dict = dict(mod.named_parameters(recurse=False))
        for p_name, p in params_dict.items():
          mod.register_parameter(p_name, nn.Parameter(p))
          nn.utils.parametrize.register_parametrization(
              mod, p_name, Parametrization(), unsafe=True
          )
      return model

    class MLPModule(nn.Module):

      def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(16, 16, bias=False)

      def forward(self, x):
        return self.net1(x)

    fw_compiler = functools.partial(
        _count_ops,
        freqs=[1, 1],
        ops=[torch.ops.aten.mul.Tensor, torch.ops.aten.sigmoid.default],
    )
    bw_compiler = functools.partial(
        _count_ops,
        freqs=[
            3 if partition_fn is _default_partition else 2,
            1,
        ],
        ops=[torch.ops.aten.mul.Tensor, torch.ops.aten.sigmoid.default],
    )
    backend = _TpuRecordingBackend(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partition_fn=partition_fn,
    )

    model = apply_parametrization(MLPModule()).to(self.device)
    model_compiled = torch.compile(
        copy.deepcopy(model), backend=backend, fullgraph=True
    )
    inp = torch.randn(8, 16, device=self.device, requires_grad=True)
    inp_compiled = inp.detach().clone().requires_grad_(True)

    out = model(inp)
    out.sum().backward()
    out_compiled = model_compiled(inp_compiled)
    out_compiled.sum().backward()

    test_utils.assert_close(out, out_compiled)
    test_utils.assert_close(inp.grad, inp_compiled.grad)

  def test_list_inputs(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_list_inputs
    class MockModule(torch.nn.Module):

      def forward(self, x, ys):
        b = torch.cos(ys[0])
        c = torch.cos(ys[1])
        return (x, [b, c])

    mod = MockModule().to(self.device)

    def fn(x, ys):
      return torch.utils.checkpoint.checkpoint(mod, x, ys, use_reentrant=True)

    x = torch.randn(4, 4, device=self.device)
    y = torch.randn(4, 4, device=self.device)
    z = torch.randn(4, 4, device=self.device)
    ref = fn(x, [y, z])
    opt_fn = torch.compile(fn, backend=_TpuRecordingBackend(), fullgraph=True)
    res = opt_fn(x, [y, z])
    test_utils.assert_close(ref[0], res[0])
    test_utils.assert_close(ref[1][0], res[1][0])
    test_utils.assert_close(ref[1][1], res[1][1])

  def test_distributed_utils_checkpoint_wrapper(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_distributed_utils_checkpoint_wrapper
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (  # pylint: disable=g-import-not-at-top
        checkpoint_wrapper as dist_checkpoint_wrapper,
    )

    class MockModule(torch.nn.Module):

      def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.c = 2

      def forward(self, x):
        x = torch.sin(x)
        x = self.linear(x)
        x = torch.cos(x)
        return x * self.c

    mod = dist_checkpoint_wrapper(MockModule().to(self.device))
    x = torch.randn(4, 4, device=self.device)
    ref = mod(x)
    opt_mod = torch.compile(mod, backend=_TpuRecordingBackend(), fullgraph=True)
    res = opt_mod(x)
    test_utils.assert_close(ref, res)

  def test_return_same_element_twice(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_return_same_element_twice
    def gn(x):
      y = torch.sin(x)
      return y, y

    def fn(x):
      return torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=True)

    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    ref = fn(x)

    backend = _TpuRecordingBackend()
    opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
    res = opt_fn(x)
    test_utils.assert_close(ref[0], res[0])
    test_utils.assert_close(ref[1], res[1])
    self.assertLen(backend.graphs, 1)
    wrap_node = _find_first_node(backend.graphs[0], _tag_activation_checkpoint)
    self.assertIsNotNone(wrap_node)

  def test_frozen_dataclass_pytree_output(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_frozen_dataclass_pytree_output
    @dataclasses.dataclass(frozen=True)
    class InputNode:
      x: torch.Tensor

    @dataclasses.dataclass(frozen=True)
    class OutputNode:
      y: torch.Tensor

    pytree.register_dataclass(InputNode)
    pytree.register_dataclass(OutputNode)

    def cleanup():
      for cls in [InputNode, OutputNode]:
        pytree.SUPPORTED_NODES.pop(cls, None)
        pytree.SUPPORTED_SERIALIZED_TYPES.pop(cls, None)
        pytree.CONSTANT_NODES.discard(cls)

    try:

      class TinyMLP(nn.Module):

        def __init__(self):
          super().__init__()
          self.fc1 = nn.Linear(4, 8)
          self.fc2 = nn.Linear(8, 4)

        def forward(self, inp: InputNode) -> OutputNode:
          h = self.fc1(inp.x)
          h = torch.nn.functional.silu(h)
          y = self.fc2(h)
          return OutputNode(y=y)

      mlp = TinyMLP().to(self.device)

      def checkpointed_forward(inp):
        return torch.utils.checkpoint.checkpoint(
            mlp.forward,
            inp,
            use_reentrant=False,
            preserve_rng_state=True,
        )

      input_eager = InputNode(
          x=torch.randn(2, 4, device=self.device, requires_grad=True)
      )
      torch.manual_seed(0)
      output_eager = checkpointed_forward(input_eager)
      output_eager.y.sum().backward()

      input_compiled = InputNode(
          x=input_eager.x.detach().clone().requires_grad_(True)
      )
      torch.manual_seed(0)
      compiled_fn = torch.compile(
          checkpointed_forward,
          fullgraph=True,
          backend=_TpuRecordingBackend(),
      )
      output_compiled = compiled_fn(input_compiled)
      output_compiled.y.sum().backward()

      test_utils.assert_close(output_eager.y, output_compiled.y)
      test_utils.assert_close(input_eager.x.grad, input_compiled.x.grad)
    finally:
      cleanup()

  def test_pre_mode_decomp_has_sac_ignored_ops(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_pre_mode_decomp_has_sac_ignored_ops
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest(
          "Exempting detach from _sac_ignored_ops during pre-mode decomposition"
          " requires PyTorch >= 2.12"
      )
    save_ops = {
        torch.ops.aten.sin.default,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.cos.default,
    }

    def policy_fn(ctx, func, *args, **kwargs):
      del ctx, args, kwargs
      if func in save_ops:
        return checkpoint.CheckpointPolicy.MUST_SAVE
      return checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    @torch._dynamo.allow_in_graph
    def op_with_detach(x):
      a = x.sin()
      out = a.detach() + a
      out = out.cos()
      return out

    self.assertEqual(
        self._get_sac_annotations(op_with_detach, policy_fn),
        "\n".join([
            "sin: aten.sin.default -> MUST_SAVE",
            "detach_1: aten.detach.default -> PREFER_RECOMPUTE",
            "add: aten.add.Tensor -> MUST_SAVE",
            "cos: aten.cos.default -> MUST_SAVE",
        ]),
    )

  def test_multi_output_op(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingViaTagsTests.test_multi_output_op
    def policy_fn(ctx, func, *args, **kwargs):
      del ctx, args, kwargs
      if func == torch.ops.aten.topk.default:
        return checkpoint.CheckpointPolicy.MUST_SAVE
      return checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    def fn(x):
      x = x.sin()
      vals, _ = torch.topk(x, k=2)
      out = vals.sum()
      out = out.cos()
      return out

    self.assertEqual(
        self._get_sac_annotations(fn, policy_fn),
        "\n".join([
            "sin: aten.sin.default -> PREFER_RECOMPUTE",
            "topk: aten.topk.default -> MUST_SAVE",
            "getitem: <built-in function getitem> -> MUST_SAVE",
            "getitem_1: <built-in function getitem> -> MUST_SAVE",
            "sum_1: aten.sum.default -> PREFER_RECOMPUTE",
            "cos: aten.cos.default -> PREFER_RECOMPUTE",
        ]),
    )


class ActivationCheckpointingSharedModuleTests(seed_test_utils.RepeatableTest):
  """Provenance: `test/dynamo/test_activation_checkpointing.py::ActivationCheckpointingSharedModuleTests`."""

  def test_sac_with_bound_method_context_fn(self):
    # Provenance: test_activation_checkpointing.py::ActivationCheckpointingSharedModuleTests.test_sac_with_bound_method_context_fn
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.15.0.
    if test_utils.get_torch_version() < (2, 15):
      self.skipTest(
          "Tracing bound method context_fn in SAC requires PyTorch >= 2.15"
      )
    device = torch.device("tpu")

    class Ctxs:

      def make(self):
        return checkpoint.create_selective_checkpoint_contexts(
            [torch.ops.aten.mm.default]
        )

    ctxs = Ctxs()

    def f(x, y):
      return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

    def fn(x, y):
      return torch.utils.checkpoint.checkpoint(
          f, x, y, use_reentrant=False, context_fn=ctxs.make
      )

    opt_fn = torch.compile(fn, backend=_TpuRecordingBackend(), fullgraph=True)
    a = torch.randn(4, 4, requires_grad=True, device=device)
    b = torch.randn(4, 4, requires_grad=True, device=device)
    test_utils.assert_close(opt_fn(a, b), fn(a, b))


class RematerializeACNodesPassTests(seed_test_utils.RepeatableTest):
  """Provenance: `test/dynamo/test_activation_checkpointing.py::RematerializeACNodesPassTests`."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def count_op(self, gm: fx.GraphModule, target: Any) -> int:
    # Provenance: test_activation_checkpointing.py::RematerializeACNodesPassTests.count_op
    return sum(1 for n in gm.graph.nodes if n.target == target)

  def _compile_and_capture(
      self,
      fn: Callable[..., Any],
      remat_using_tags_for_fwd_loss_bwd_graph: bool,
      inputs: tuple[torch.Tensor, ...],
  ) -> tuple[Any, fx.GraphModule]:
    # Provenance: test_activation_checkpointing.py::RematerializeACNodesPassTests._compile_and_capture
    captured_gm: list[fx.GraphModule] = []
    if isinstance(fn, nn.Module):
      fn = copy.deepcopy(fn).to(self.device)
    tpu_inputs = tuple(
        t.to(self.device).detach().clone().requires_grad_(t.requires_grad)
        for t in inputs
    )

    static_compiler_cls = (
        torch_tpu_compile._backend.compiler.StaticCompiler  # pylint: disable=protected-access
    )
    split_compiler_mod = (
        torch_tpu_compile._backend.split_compiler  # pylint: disable=protected-access
    )
    orig_compile = static_compiler_cls._compile  # pylint: disable=protected-access
    orig_split_module = split_compiler_mod.split_module

    def wrapped_compile(self_compiler, graph_module, *args, **kwargs):
      captured_gm.append(graph_module)
      return orig_compile(self_compiler, graph_module, *args, **kwargs)

    torch._dynamo.reset()
    with (
        mock.patch.object(
            split_compiler_mod,
            "split_module",
            side_effect=functools.partial(
                orig_split_module, keep_original_node_name=True
            ),
        ),
        mock.patch.object(
            static_compiler_cls,
            "_compile",
            autospec=True,
            side_effect=wrapped_compile,
        ),
        torch._functorch.config.patch(
            remat_using_tags_for_fwd_loss_bwd_graph=remat_using_tags_for_fwd_loss_bwd_graph
        ),
        torch._dynamo.config.patch(trace_autograd_ops=True),
    ):
      compiled_fn = torch.compile(
          fn, backend="tpu", fullgraph=True, options={"serializable": False}
      )
      result = compiled_fn(*tpu_inputs)

    self.assertNotEmpty(captured_gm)
    return result, captured_gm[-1]

  def test_ac_rematerialize_simple_forward_backward(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_ac_rematerialize_simple_forward_backward
    # Modified from upstream: also asserts optimization barrier count on `gm_with`/`gm_without`.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not tag
    # duplicated remat nodes with `recompute`/`ac_graph_id`/`autograd_backward`
    # (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    y = torch.randn(4, 4, device=self.device, requires_grad=True)

    def simple_fwd_bwd(x, y):
      z = torch.utils.checkpoint.checkpoint(
          lambda a, b: torch.sigmoid(torch.matmul(a, b)),
          x,
          y,
          use_reentrant=False,
      )
      loss = z.sum()
      dx, dy = _grad(loss, (x, y))
      return dx.detach(), dy.detach()

    (dx1, dy1), gm_without = self._compile_and_capture(
        simple_fwd_bwd, False, (x, y)
    )
    (dx2, dy2), gm_with = self._compile_and_capture(
        simple_fwd_bwd, True, (x, y)
    )

    test_utils.assert_close(dx1, dx2)
    test_utils.assert_close(dy1, dy2)

    mm_with = self.count_op(gm_with, torch.ops.aten.mm.default)
    mm_without = self.count_op(gm_without, torch.ops.aten.mm.default)
    sigmoid_with = self.count_op(gm_with, torch.ops.aten.sigmoid.default)
    sigmoid_without = self.count_op(gm_without, torch.ops.aten.sigmoid.default)
    self.assertEqual(mm_with, 4, "mm should be recomputed in backward")
    self.assertEqual(mm_without, 3)
    self.assertEqual(
        sigmoid_with, 2, "sigmoid should be recomputed in backward"
    )
    self.assertEqual(sigmoid_without, 1)
    self.assertLen(_get_barriers(gm_with), 1)
    self.assertEmpty(_get_barriers(gm_without))

  def test_ac_rematerialize_with_rng_ops_raises_error(self):
    # Provenance: test_activation_checkpointing.py::RematerializeACNodesPassTests.test_ac_rematerialize_with_rng_ops_raises_error
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    x = torch.randn(4, 4, device=self.device, requires_grad=True)

    def fwd_bwd_with_rng(x):
      z = torch.utils.checkpoint.checkpoint(
          lambda a: torch.sigmoid(a + torch.rand_like(a)),
          x,
          use_reentrant=False,
      )
      loss = z.sum()
      dx = _grad(loss, x)[0]
      return dx

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Ported upstream error test.
        torch._dynamo.exc.BackendCompilerFailed,
        "Activation checkpoint rematerialization in `forward-loss-backward`"
        " graph does not support RNG ops in recompute regions.",
    ):
      self._compile_and_capture(fwd_bwd_with_rng, True, (x,))

  def test_ac_rematerialize_with_sdpa_dropout_zero(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_ac_rematerialize_with_sdpa_dropout_zero
    # Modified from upstream: also asserts optimization barrier count on `gm_with`.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not tag
    # duplicated remat nodes with `recompute`/`ac_graph_id`/`autograd_backward`
    # (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    sdpa_ops = {
        torch.ops.aten.scaled_dot_product_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,  # pylint: disable=protected-access
        torch.ops.aten._scaled_dot_product_flash_attention.default,  # pylint: disable=protected-access
        torch.ops.aten._scaled_dot_product_efficient_attention.default,  # pylint: disable=protected-access
        torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,  # pylint: disable=protected-access
        torch.ops.aten._flash_attention_forward.default,  # pylint: disable=protected-access
    }

    def policy_fn(ctx, op, *args, **kwargs):
      del ctx, args, kwargs
      if op in sdpa_ops:
        return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
      return torch.utils.checkpoint.CheckpointPolicy.PREFER_SAVE

    context_fn = functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
    )

    q = torch.randn(2, 4, 128, 64, device=self.device, requires_grad=True)
    k = torch.randn(2, 4, 128, 64, device=self.device, requires_grad=True)
    v = torch.randn(2, 4, 128, 64, device=self.device, requires_grad=True)

    def fwd_bwd_with_sdpa(q, k, v):
      z = torch.utils.checkpoint.checkpoint(
          lambda q, k, v: F.scaled_dot_product_attention(
              q, k, v, dropout_p=0.0
          ),
          q,
          k,
          v,
          use_reentrant=False,
          context_fn=context_fn,
      )
      loss = z.sum()
      dq, dk, dv = _grad(loss, (q, k, v))
      return z.detach(), dq, dk, dv

    result_with, gm_with = self._compile_and_capture(
        fwd_bwd_with_sdpa, True, (q, k, v)
    )
    result_without, _ = self._compile_and_capture(
        fwd_bwd_with_sdpa, False, (q, k, v)
    )
    eager_inputs = tuple(
        t.detach().clone().requires_grad_(True) for t in (q, k, v)
    )
    result_eager = fwd_bwd_with_sdpa(*eager_inputs)

    for actual, expected in zip(result_with, result_without):
      test_utils.assert_close(actual, expected)
    for actual, expected in zip(result_with, result_eager):
      test_utils.assert_close(actual, expected)
    self.assertLen(_get_barriers(gm_with), 1)

  def test_ac_rematerialize_with_no_annotations(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_ac_rematerialize_with_no_annotations
    # Modified from upstream: also asserts optimization barrier count on `gm_with`.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not tag
    # duplicated remat nodes with `recompute`/`ac_graph_id`/`autograd_backward`
    # (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    x = torch.randn(4, 4, device=self.device, requires_grad=True)

    def fwd_bwd(x):
      z = torch.utils.checkpoint.checkpoint(
          lambda a: torch.sigmoid(a + 4), x, use_reentrant=False
      )
      loss = z.sum()
      return _grad(loss, x)[0]

    result_with, gm_with = self._compile_and_capture(fwd_bwd, True, (x,))
    result_without, gm_without = self._compile_and_capture(fwd_bwd, False, (x,))

    test_utils.assert_close(result_with, result_without)
    sigmoid_with = self.count_op(gm_with, torch.ops.aten.sigmoid.default)
    sigmoid_without = self.count_op(gm_without, torch.ops.aten.sigmoid.default)
    self.assertEqual(
        sigmoid_with, 2, "sigmoid should be recomputed in backward"
    )
    self.assertEqual(sigmoid_without, 1)
    self.assertLen(_get_barriers(gm_with), 1)

  def test_ac_rematerialize_with_selective_checkpoint_policy(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_ac_rematerialize_with_selective_checkpoint_policy
    # Modified from upstream: also asserts optimization barrier count on `gm_with`.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not tag
    # duplicated remat nodes with `recompute`/`ac_graph_id`/`autograd_backward`
    # (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    x = torch.randn(4, 128, device=self.device, requires_grad=True)
    w1 = torch.randn(128, 128, device=self.device, requires_grad=True)
    b1 = torch.randn(128, device=self.device, requires_grad=True)

    def policy_fn(ctx, op, *args, **kwargs):
      del ctx, args, kwargs
      if op == torch.ops.aten.addmm.default:
        return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
      return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

    context_fn = functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
    )

    def fwd_bwd_with_policy(x, w1, b1):
      def checkpoint_fn(inp, w, b):
        linear = torch.nn.functional.linear(inp, w, b)
        return torch.relu(linear)

      result = torch.utils.checkpoint.checkpoint(
          checkpoint_fn, x, w1, b1, use_reentrant=False, context_fn=context_fn
      )
      loss = result.sum()
      dx, dw, db = _grad(loss, (x, w1, b1))
      return dx, dw, db

    result_with, gm_with = self._compile_and_capture(
        fwd_bwd_with_policy, True, (x, w1, b1)
    )
    result_without, gm_without = self._compile_and_capture(
        fwd_bwd_with_policy, False, (x, w1, b1)
    )

    for a, b in zip(result_with, result_without):
      test_utils.assert_close(a, b)

    addmm_without = self.count_op(gm_without, torch.ops.aten.addmm.default)
    relu_without = self.count_op(gm_without, torch.ops.aten.relu.default)
    addmm_with = self.count_op(gm_with, torch.ops.aten.addmm.default)
    relu_with = self.count_op(gm_with, torch.ops.aten.relu.default)

    self.assertEqual(addmm_without, addmm_with)
    self.assertEqual(relu_with, relu_without + 1)

    recomputed_nodes = [
        n.name for n in gm_with.graph.nodes if "_recomputed" in n.name
    ]
    self.assertNotIn("addmm_recomputed", recomputed_nodes)
    self.assertTrue(any("relu" in name for name in recomputed_nodes))
    self.assertLen(_get_barriers(gm_with), 1)

  def test_multiple_user_phase_annotations_errors(self):
    # Provenance: test_activation_checkpointing.py::RematerializeACNodesPassTests.test_multiple_user_phase_annotations_errors
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    w = torch.randn(4, 4, device=self.device, requires_grad=True)

    def fn(x, w):
      z = torch.utils.checkpoint.checkpoint(
          lambda a, b: torch.sin(a @ b), x, w, use_reentrant=False
      )
      loss = z.sum()
      with torch.fx.traceback.annotate({"phase": "backward"}):
        dx, dw = _grad(loss, (x, w))
      out = dx + dw
      with torch.fx.traceback.annotate({"phase": "backward"}):
        out = out * 2
      return out.detach()

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Ported upstream error test.
        RuntimeError, "backward regions annotated"
    ):
      self._compile_and_capture(fn, True, (x, w))

  def test_user_phase_annotation_with_extra_autograd_grad(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_user_phase_annotation_with_extra_autograd_grad
    # Modified from upstream: also asserts optimization barrier count on `gm_with`.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not tag
    # duplicated remat nodes with `recompute`/`ac_graph_id`/`autograd_backward`
    # (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    x = torch.randn(4, 4, device=self.device, requires_grad=True)
    w1 = torch.randn(4, 4, device=self.device, requires_grad=True)
    w2 = torch.randn(4, 4, device=self.device, requires_grad=True)

    def fn(x, w1, w2):
      z1 = torch.utils.checkpoint.checkpoint(
          lambda a, b: torch.sin(a @ b), x, w1, use_reentrant=False
      )
      z2 = torch.utils.checkpoint.checkpoint(
          lambda a, b: torch.sigmoid(a @ b), x, w2, use_reentrant=False
      )
      loss1 = z1.sum()
      loss2 = z2.sum()
      with torch.fx.traceback.annotate({"phase": "backward"}):
        dx1 = _grad(loss1, (x,))
      dx2 = _grad(loss2, (x,))
      return (dx1[0] + dx2[0]).detach()

    result_with, gm_with = self._compile_and_capture(fn, True, (x, w1, w2))
    result_without, _ = self._compile_and_capture(fn, False, (x, w1, w2))
    test_utils.assert_close(result_with, result_without)
    self.assertLen(_get_barriers(gm_with), 1)

  def test_chunked_loss_remat(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_chunked_loss_remat
    # Modified from upstream: also asserts optimization barrier count on `gm_with`.
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not
    # automatically detect traced `loss.backward()` nodes
    # (`autograd_backward=True`, pytorch/pytorch#179105) or tag duplicated
    # remat nodes (pytorch/pytorch#181937).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    dim = 32
    chunksz = 4

    class Block(nn.Module):

      def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=False)
        self.l2 = nn.Linear(dim, dim, bias=False)

      def _fn(self, x):
        return self.l2(F.gelu(self.l1(x), approximate="tanh"))

      def forward(self, x):
        return checkpoint.checkpoint(self._fn, x, use_reentrant=False)

    class ChunkedLoss(nn.Module):

      def __init__(self):
        super().__init__()
        self.block = Block()
        self.head = nn.Linear(dim, dim, bias=False)

      def forward(self, x, y):
        x = self.block(x)
        x_detached = x.detach().requires_grad_()
        total = 0
        for start in range(0, x_detached.shape[0], chunksz):
          end = start + chunksz
          chunk_loss = (
              F.mse_loss(
                  self.head(x_detached[start:end]),
                  y[start:end],
                  reduction="sum",
              )
              / x_detached.shape[0]
          )
          chunk_loss.backward()
          total = total + chunk_loss.detach()
        x.backward(x_detached.grad)
        return total

    model = ChunkedLoss().to(self.device)
    x = torch.randn(12, dim, device=self.device)
    y = torch.randn(12, dim, device=self.device)

    result_with, gm_with = self._compile_and_capture(model, True, (x, y))
    result_without, gm_without = self._compile_and_capture(model, False, (x, y))

    test_utils.assert_close(result_with, result_without)
    gelu_without = self.count_op(gm_without, torch.ops.aten.gelu.default)
    gelu_with = self.count_op(gm_with, torch.ops.aten.gelu.default)
    self.assertEqual(gelu_without, 1)
    self.assertEqual(gelu_with, 2, "gelu should be recomputed in backward")
    self.assertLen(_get_barriers(gm_with), 1)

  def test_two_backward_regions_needing_remat_errors(self):
    # Provenance: test_activation_checkpointing.py::RematerializeACNodesPassTests.test_two_backward_regions_needing_remat_errors
    # TODO: b/576283069 - Remove skip once minimum OSS PyTorch is >= 2.12.0.
    # In PyTorch < 2.12, `remat_using_tags_for_fwd_loss_bwd_graph` did not
    # automatically detect traced `loss.backward()` nodes
    # (`autograd_backward=True`, pytorch/pytorch#179105).
    if test_utils.get_torch_version() < (2, 12):
      self.skipTest("Fused-step AC rematerialization requires PyTorch >= 2.12")
    dim = 32

    class Block(nn.Module):

      def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=False)
        self.l2 = nn.Linear(dim, dim, bias=False)

      def _fn(self, x):
        return self.l2(F.gelu(self.l1(x), approximate="tanh"))

      def forward(self, x):
        return checkpoint.checkpoint(self._fn, x, use_reentrant=False)

    class TwoBackwards(nn.Module):

      def __init__(self):
        super().__init__()
        self.block1 = Block()
        self.block2 = Block()

      def forward(self, x):
        y = self.block1(x)
        z = self.block2(y)
        z.sum().backward(retain_graph=True)
        y.sum().backward()
        return z.detach()

    x = torch.randn(8, dim, device=self.device, requires_grad=True)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Ported upstream error test.
        torch._dynamo.exc.BackendCompilerFailed,
        "require recomputation",
    ):
      self._compile_and_capture(TwoBackwards(), True, (x,))


if __name__ == "__main__":
  absltest.main()
