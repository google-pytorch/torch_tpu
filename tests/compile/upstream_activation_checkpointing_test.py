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
import functools
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import fx
from torch import nn
import torch.nn.functional as F
from torch.utils import checkpoint
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


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


class _TpuAotAutogradBackend:
  """Wraps `TpuBackend` with custom `aot_autograd` compiler/partitioner hooks.

  Upstream `test/dynamo/test_activation_checkpointing.py` passes custom
  `aot_autograd(fw_compiler=..., bw_compiler=..., partition_fn=...,
  decompositions=...)` backends to `torch.compile` to inspect the partitioned
  forward/backward FX graphs (e.g. via `_count_ops`) or override the partitioner
  and decomposition table. Passing `aot_autograd(...)` directly as the Dynamo
  backend executes the partitioned FX graphs op-by-op in eager mode and bypasses
  `TpuBackend`, whereas `TpuBackend` hardcodes its own internal `aot_autograd`
  invocation (`_compile_graph_module`, `min_cut_rematerialization_partition`,
  and `_TPU_DECOMPOSITIONS`) without exposing those hooks as options. This
  adapter runs the full `TpuBackend` compilation pipeline while injecting the
  test's custom `aot_autograd` callbacks, partitioner, and decompositions.
  """

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

  def __call__(
      self,
      gm: fx.GraphModule,
      example_inputs: list[torch.Tensor],
      **kwargs: Any,
  ) -> Callable[..., Any]:
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
      user_compiler = self._fw_compiler if is_fwd else self._bw_compiler
      if user_compiler is not None:
        user_compiler(graph_module, inputs)
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
      if self._fw_compiler is not None or self._bw_compiler is not None:
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
  ) -> Any:
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

    debugs: list[Any] = []
    options = (
        {"serializable": False, "debug_callback": debugs.append}
        if backend == "tpu"
        else None
    )
    torch.manual_seed(0)
    compiled_fn = torch.compile(
        cloned_fn, fullgraph=fullgraph, backend=backend, options=options
    )
    result = compiled_fn(*cloned_args)
    result.sum().backward()

    if not skip_check:
      test_utils.assert_close(result, expected)
      for arg, cloned_arg in zip(tpu_args, cloned_args):
        test_utils.assert_close(arg.grad, cloned_arg.grad)
    return debugs[-1] if debugs else None

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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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

    backend = _TpuAotAutogradBackend(partition_fn=partition_fn)
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

    backend = _TpuAotAutogradBackend(partition_fn=partition_fn)
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    debug = self._validate(fn, "tpu", x, y, skip_check=True)
    self.assertIsNotNone(debug)
    self.assertEqual(
        debug.stablehlo_backward_text[0].count(
            "stablehlo.optimization_barrier"
        ),
        2,
    )

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
    self._validate(fn, "tpu", x, y)

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
    debug = self._validate(fn, "tpu", x, skip_check=True)
    self.assertIsNotNone(debug)
    self.assertEqual(
        debug.stablehlo_backward_text[0].count(
            "stablehlo.optimization_barrier"
        ),
        1,
    )

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

    cnt = torch._dynamo.testing.CompileCounterWithBackend("tpu")
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
      backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
    backend = _TpuAotAutogradBackend(
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
      backend = _TpuAotAutogradBackend(
          fw_compiler=fw_compiler,
          bw_compiler=bw_compiler,
          partition_fn=partition_fn,
      )
      self._validate(fn, backend, x, skip_check=True)

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
    opt_fn = torch.compile(
        fn, backend="tpu", fullgraph=True, options={"serializable": False}
    )
    res = opt_fn(x, [y, z])
    test_utils.assert_close(ref[0], res[0])
    test_utils.assert_close(ref[1][0], res[1][0])
    test_utils.assert_close(ref[1][1], res[1][1])


class RematerializeACNodesPassTests(seed_test_utils.RepeatableTest):
  """Provenance: `test/dynamo/test_activation_checkpointing.py::RematerializeACNodesPassTests`."""

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _compile_and_capture(
      self,
      fn: Callable[..., Any],
      remat_using_tags_for_fwd_loss_bwd_graph: bool,
      inputs: tuple[torch.Tensor, ...],
  ) -> tuple[Any, Any]:
    # Provenance: test_activation_checkpointing.py::RematerializeACNodesPassTests._compile_and_capture
    if isinstance(fn, nn.Module):
      fn = copy.deepcopy(fn).to(self.device)
    tpu_inputs = tuple(
        t.to(self.device).detach().clone().requires_grad_(t.requires_grad)
        for t in inputs
    )

    debugs: list[Any] = []
    torch._dynamo.reset()
    with (
        torch._functorch.config.patch(
            remat_using_tags_for_fwd_loss_bwd_graph=remat_using_tags_for_fwd_loss_bwd_graph
        ),
        torch._dynamo.config.patch(trace_autograd_ops=True),
    ):
      compiled_fn = torch.compile(
          fn,
          backend="tpu",
          fullgraph=True,
          options={"serializable": False, "debug_callback": debugs.append},
      )
      result = compiled_fn(*tpu_inputs)

    self.assertNotEmpty(debugs)
    return result, debugs[-1]

  def test_ac_rematerialize_with_sdpa_dropout_zero(self):
    # Provenance (modified): test_activation_checkpointing.py::RematerializeACNodesPassTests.test_ac_rematerialize_with_sdpa_dropout_zero
    # Modified from upstream: also asserts optimization barrier count in compiled StableHLO.
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

    result_with, debug_with = self._compile_and_capture(
        fwd_bwd_with_sdpa, True, (q, k, v)
    )
    result_without, debug_without = self._compile_and_capture(
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
    self.assertEqual(
        debug_with.stablehlo_forward_text[0].count(
            "stablehlo.optimization_barrier"
        ),
        1,
    )
    self.assertEqual(
        debug_without.stablehlo_forward_text[0].count(
            "stablehlo.optimization_barrier"
        ),
        0,
    )


if __name__ == "__main__":
  absltest.main()
