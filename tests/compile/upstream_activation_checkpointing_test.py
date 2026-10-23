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

_PARTITION_FN_PARAMS = (
    ("min_cut", _min_cut_partition),
    ("default", _default_partition),
)


def _grad(*args, **kwargs):
  # Provenance: test_activation_checkpointing.py::_grad
  return torch.autograd.grad(*args, **kwargs)


def _count_ops(
    gm: fx.GraphModule,
    inputs: Any,
    *,
    freqs: list[int],
    ops: list[Any],
) -> fx.GraphModule:
  # Provenance: test_activation_checkpointing.py::count_ops
  del inputs

  def match_rng_op(node: fx.Node, target_op: Any) -> bool:
    if isinstance(node.target, torch._ops.HigherOrderOperator):  # pylint: disable=protected-access
      if node.name == "run_and_save_rng_state":
        return node.args[0] == target_op
      elif node.name == "run_with_rng_state":
        return node.args[1] == target_op
      elif node.name == "graphsafe_run_with_rng_state":
        return node.args[0] == target_op
    return False

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
  return gm


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
  ):
    self._fw_compiler = fw_compiler
    self._bw_compiler = bw_compiler
    self._partition_fn = partition_fn
    self.bw_graphs: list[fx.GraphModule] = []

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
      if is_fwd:
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
    compiled_fn = torch.compile(cloned_fn, fullgraph=True, backend=backend)
    result = compiled_fn(*cloned_args)
    result.sum().backward()

    if not skip_check:
      test_utils.assert_close(result, expected)
      for arg, cloned_arg in zip(tpu_args, cloned_args):
        test_utils.assert_close(arg.grad, cloned_arg.grad)

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


if __name__ == "__main__":
  absltest.main()
