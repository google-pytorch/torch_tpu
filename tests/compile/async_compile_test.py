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

"""Unit tests for experimental asynchronous compilation and warmup overlapping."""

from collections.abc import Callable
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd import autograd_cache
from torch._inductor.codecache import FxGraphCache
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import clear_caches
from torch.compiler._cache import CacheArtifactManager
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import compiler as tpu_compiler_mod
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils


class SimpleModel(nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + 1.0


class AnotherModel(nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * 2.0


class KwargModel(nn.Module):

  def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


class TrainableModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.w = nn.Parameter(torch.randn(8, 8, device="tpu"))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, self.w)


class MultiOutputModel(nn.Module):

  def forward(
      self, x: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    return x + 1.0, x * 2.0, {"sub": x - 3.0}


class StochasticModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.dropout = nn.Dropout(p=0.5)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.dropout(x) + torch.randn_like(x)


class PrematureResolutionError(AssertionError):
  """Raised when an AsyncCompiledArtifact is resolved before the explicit barrier."""


class CompileBarrier:
  """Deterministic synchronization controller for compiler unit tests."""

  def __init__(self):
    self.compile_started_events: dict[int, threading.Event] = {}
    self.allow_finish_events: dict[int, threading.Event] = {}
    self.compile_finished_events: dict[int, threading.Event] = {}
    self._lock = threading.Lock()
    self.call_count = 0

  def _get_event(
      self, event_dict: dict[int, threading.Event], index: int
  ) -> threading.Event:
    with self._lock:
      if index not in event_dict:
        event_dict[index] = threading.Event()
      return event_dict[index]

  def intercept_compile(
      self, orig_compile_fn: Callable[..., Any]
  ) -> Callable[..., Any]:
    """Wraps StaticCompiler._compile to coordinate with the main test thread."""
    barrier = self

    def wrapped(self_compiler, *args, **kwargs):
      with barrier._lock:
        idx = barrier.call_count
        barrier.call_count += 1

      started = barrier._get_event(barrier.compile_started_events, idx)
      allow_finish = barrier._get_event(barrier.allow_finish_events, idx)
      finished = barrier._get_event(barrier.compile_finished_events, idx)

      started.set()
      allow_finish.wait(timeout=10.0)  # Fails fast on deadlock
      try:
        result = orig_compile_fn(self_compiler, *args, **kwargs)
      finally:
        finished.set()
      return result

    return wrapped

  def wait_for_compile_start(self, index: int, timeout: float = 5.0) -> bool:
    return self._get_event(self.compile_started_events, index).wait(
        timeout=timeout
    )

  def release_compile(self, index: int) -> None:
    self._get_event(self.allow_finish_events, index).set()

  def release_all(self) -> None:
    with self._lock:
      for event in self.allow_finish_events.values():
        event.set()


class ResolutionCanary:
  """Monitors AsyncCompiledArtifact instances to guarantee zero premature resolution."""

  def __init__(self):
    self.registered_artifacts: list[Any] = []
    self.resolution_allowed = False

  def watch(self, artifact: Any) -> Any:
    self.registered_artifacts.append(artifact)
    orig_resolve = artifact.resolve

    def strict_resolve():
      if not self.resolution_allowed:
        raise PrematureResolutionError(
            f"Artifact {artifact} was resolved prematurely! "
            "A PyTorch wrapper or Dynamo guard accessed an attribute that"
            " triggered _resolve()."
        )
      return orig_resolve()

    artifact.resolve = strict_resolve
    return artifact

  def allow_resolution(self):
    self.resolution_allowed = True


def _reset() -> None:
  """Resets compiler, dynamo, inductor, and AOTAutograd caches."""
  if (
      hasattr(torch.fx, "traceback")
      and getattr(torch.fx.traceback, "current_meta", None) is not None
  ):
    torch.fx.traceback.current_meta.clear()
  torch.compiler.reset()
  torch.tpu._clear_cache()
  torch._dynamo.reset()
  DynamoCache.clear()
  PrecompileContext.clear()
  counters.clear()
  autograd_cache.AOTAutogradCache.clear()
  CacheArtifactManager.clear()
  PyCodeCache.cache_clear(purge=True)
  FxGraphCache.clear()
  clear_caches()


class AsyncCompileTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    _reset()

  def tearDown(self):
    _reset()
    super().tearDown()

  def test_async_compile_basic(self):
    model = SimpleModel().to("tpu")
    bucket_sizes = [4, 8, 16]
    warmup_inputs = [(torch.randn(b, 10, device="tpu"),) for b in bucket_sizes]

    compiled_model = tpu_compile.async_compile(
        model,
        warmup_inputs,
    )
    self.assertTrue(callable(compiled_model))

    x = torch.ones(8, 10, device="tpu")
    res = compiled_model(x)
    self.assertEqual(res.shape, (8, 10))
    test_utils.assert_close(res.cpu(), (x + 1.0).cpu())

  def test_async_compile_deterministic_overlap(self):
    """Deterministically proves Shape 2 tracing overlaps with Shape 1 compilation."""
    barrier = CompileBarrier()
    model = SimpleModel().to("tpu")

    shape1 = (torch.randn(4, 16, device="tpu"),)
    shape2 = (torch.randn(8, 16, device="tpu"),)

    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler,
        "_compile",
        barrier.intercept_compile(tpu_compiler_mod.StaticCompiler._compile),
    ):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
          options={"async_compile": True},
      )

      with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm1:
        compiled(*shape1)
      art1 = cm1.exception.artifact
      self.assertIsNotNone(art1)
      self.assertFalse(art1.is_resolved)

      self.assertTrue(barrier.wait_for_compile_start(0))

      with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm2:
        compiled(*shape2)
      art2 = cm2.exception.artifact
      self.assertIsNotNone(art2)

      self.assertFalse(
          art1.is_resolved,
          "Shape 1 resolved prematurely during Shape 2 tracing!",
      )
      self.assertFalse(art2.is_resolved, "Shape 2 resolved prematurely!")

      barrier.release_all()

      resolved = tpu_compile.resolve_compilations([art1, art2])
      self.assertEqual(resolved, 2)
      self.assertTrue(art1.is_resolved)
      self.assertTrue(art2.is_resolved)

      out1 = compiled(*shape1)
      out2 = compiled(*shape2)
      test_utils.assert_close(out1.cpu(), (shape1[0] + 1.0).cpu())
      test_utils.assert_close(out2.cpu(), (shape2[0] + 1.0).cpu())

  def test_async_compile_without_torch_cache_key_does_not_block(self):
    # With torch's autograd cache disabled the backend builds the bundled entry
    # under its own fallback key; building it must not resolve the compile.
    barrier = CompileBarrier()
    model = SimpleModel().to("tpu")
    x = (torch.randn(4, 16, device="tpu"),)
    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler,
        "_compile",
        barrier.intercept_compile(tpu_compiler_mod.StaticCompiler._compile),
    ), functorch_config.patch(enable_autograd_cache=False):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
          options={"async_compile": True},
      )
      with self.assertRaises(  # ASSERT_RAISES_OK=AsyncCompilationSubmitted warmup signal.
          tpu_compile.AsyncCompilationSubmitted
      ) as cm:
        compiled(*x)
      art = cm.exception.artifact
      self.assertTrue(barrier.wait_for_compile_start(0))
      self.assertFalse(art.is_resolved)
      barrier.release_all()
      self.assertEqual(tpu_compile.resolve_compilations([art]), 1)
      test_utils.assert_close(compiled(*x).cpu(), (x[0] + 1.0).cpu())

  def test_async_compile_zero_premature_resolution_canary(self):
    """Ensures no PyTorch or backend wrapper triggers premature artifact resolution."""
    canary = ResolutionCanary()
    barrier = CompileBarrier()
    model = AnotherModel().to("tpu")

    bucket_shapes = [(torch.randn(2**i, 8, device="tpu"),) for i in range(2, 5)]

    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler,
        "_compile",
        barrier.intercept_compile(tpu_compiler_mod.StaticCompiler._compile),
    ):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
          options={"async_compile": True},
      )

      artifacts = []
      for shape in bucket_shapes:
        with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm:
          compiled(*shape)
        art = canary.watch(cm.exception.artifact)
        artifacts.append(art)
        for a in artifacts:
          self.assertFalse(a.is_resolved)

      barrier.release_all()
      canary.allow_resolution()

      resolved = tpu_compile.resolve_compilations(artifacts)
      self.assertEqual(resolved, len(bucket_shapes))
      for a in artifacts:
        self.assertTrue(a.is_resolved)

  def test_async_compile_training_autograd_overlap(self):
    """Tests async compilation with autograd (forward + backward graph)."""
    model = TrainableModel()
    warmup_inputs = [
        (torch.randn(2, 8, device="tpu", requires_grad=True),),
        (torch.randn(4, 8, device="tpu", requires_grad=True),),
    ]

    compiled = tpu_compile.async_compile(
        model,
        warmup_inputs,
        no_grad=False,
    )

    x = torch.randn(4, 8, device="tpu", requires_grad=True)
    out = compiled(x)
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(model.w.grad)
    self.assertEqual(x.grad.shape, (4, 8))
    self.assertEqual(model.w.grad.shape, (8, 8))

  def test_async_compile_autograd_overlap_traceback_isolation(self):
    """Verifies autograd overlap preserves FX traceback metadata."""
    model = TrainableModel()
    warmup_inputs = [
        (torch.randn(2, 8, device="tpu", requires_grad=True),),
        (torch.randn(4, 8, device="tpu", requires_grad=True),),
    ]

    try:
      # Pre-set autograd traceback state to simulate active autograd context.
      torch.fx.traceback.should_preserve_node_meta = True
      torch.fx.traceback.set_grad_fn_seq_nr(1)
      # Compiling warmup inputs asynchronously in background threads must not
      # corrupt current_meta or wipe in_grad_fn.
      compiled = tpu_compile.async_compile(
          model,
          warmup_inputs,
          no_grad=False,
      )
      # Resetting the grad_fn_seq_nr must succeed without AssertionError.
      torch.fx.traceback.reset_grad_fn_seq_nr()
    finally:
      torch.fx.traceback.should_preserve_node_meta = False
      if getattr(torch.fx.traceback, "current_meta", None) is not None:
        torch.fx.traceback.current_meta.clear()

    x = torch.randn(4, 8, device="tpu", requires_grad=True)
    out = compiled(x)
    loss = out.sum()
    loss.backward()

    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(model.w.grad)
    self.assertEqual(x.grad.shape, (4, 8))
    self.assertEqual(model.w.grad.shape, (8, 8))

  def test_async_compile_complex_pytree_outputs(self):
    """Tests async compilation with multi-tensor and dictionary pytree outputs."""
    model = MultiOutputModel().to("tpu")
    warmup_inputs = [
        (torch.randn(2, 4, device="tpu"),),
        (torch.randn(6, 4, device="tpu"),),
    ]

    compiled = tpu_compile.async_compile(model, warmup_inputs)

    test_x = torch.ones(6, 4, device="tpu")
    add_res, mul_res, dict_res = compiled(test_x)

    test_utils.assert_close(add_res.cpu(), torch.ones(6, 4) * 2.0)
    test_utils.assert_close(mul_res.cpu(), torch.ones(6, 4) * 2.0)
    test_utils.assert_close(dict_res["sub"].cpu(), torch.ones(6, 4) * (-2.0))

  def test_async_compile_rng_generator_state(self):
    """Tests that async compile correctly handles RNG generator state updates."""
    model = StochasticModel().to("tpu")
    warmup_inputs = [
        (torch.ones(4, 16, device="tpu"),),
        (torch.ones(8, 16, device="tpu"),),
    ]

    compiled = tpu_compile.async_compile(model, warmup_inputs, no_grad=True)

    torch.manual_seed(1234)
    res1 = compiled(torch.ones(4, 16, device="tpu"))

    torch.manual_seed(1234)
    res2 = compiled(torch.ones(4, 16, device="tpu"))

    test_utils.assert_close(res1.cpu(), res2.cpu())

  def test_async_compile_error_propagation(self):
    """Verifies that background compilation failures surface cleanly during resolution."""
    model = SimpleModel().to("tpu")

    def failing_compile(*args, **kwargs):
      raise RuntimeError("Simulated XLA lowering failure")

    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler, "_compile", failing_compile
    ):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
          options={"async_compile": True},
      )

      with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm:
        compiled(torch.ones(4, 4, device="tpu"))

      art = cm.exception.artifact
      self.assertIsNotNone(art)

      with self.assertRaisesRegex(
          RuntimeError, "Simulated XLA lowering failure"
      ):
        art.resolve()

  def test_async_compile_cache_hit_no_recompile(self):
    """Verifies that repeated invocations on warmed-up shapes do not trigger recompilation."""
    model = SimpleModel().to("tpu")
    warmup_inputs = [(torch.randn(4, 8, device="tpu"),)]

    compiled = tpu_compile.async_compile(model, warmup_inputs)

    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler, "__call__"
    ) as mock_compiler:
      res1 = compiled(torch.ones(4, 8, device="tpu"))
      res2 = compiled(torch.ones(4, 8, device="tpu") * 2)

      mock_compiler.assert_not_called()
      self.assertEqual(res1.shape, (4, 8))
      self.assertEqual(res2.shape, (4, 8))

  def test_async_compile_no_grad_option(self):
    model = SimpleModel().to("tpu")
    warmup_inputs = [(torch.randn(2, 5, device="tpu", requires_grad=True),)]

    compiled_model = tpu_compile.async_compile(
        model,
        warmup_inputs,
        no_grad=False,
    )

    x = torch.ones(2, 5, device="tpu", requires_grad=True)
    res = compiled_model(x)
    self.assertTrue(res.requires_grad)

  def test_async_compile_keyword_args(self):
    model = KwargModel().to("tpu")
    warmup_inputs = [
        ((torch.ones(2, 2, device="tpu"),), {"scale": 3.0}),
        {"x": torch.ones(4, 4, device="tpu"), "scale": 5.0},
    ]

    compiled_model = tpu_compile.async_compile(
        model,
        warmup_inputs,
    )

    res1 = compiled_model(torch.ones(2, 2, device="tpu"), scale=3.0)
    test_utils.assert_close(res1.cpu(), torch.ones(2, 2) * 3.0)

  def test_async_compilation_submitted_signal(self):
    model = AnotherModel().to("tpu")
    compiled_model = torch.compile(
        torch.no_grad()(model),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
        options={"async_compile": True},
    )

    x = torch.ones(7, 13, device="tpu")
    with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm:
      compiled_model(x)

    self.assertIsNotNone(cm.exception.artifact)
    cm.exception.resolve()

    res = compiled_model(x)
    self.assertEqual(res.shape, (7, 13))
    test_utils.assert_close(res.cpu(), (x * 2.0).cpu())

  def test_resolve_compilations_helper(self):
    model = AnotherModel().to("tpu")
    compiled_model = torch.compile(
        torch.no_grad()(model),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
        options={"async_compile": True},
    )

    x = torch.ones(3, 7, device="tpu")
    with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm:
      compiled_model(x)

    resolved_count = tpu_compile.resolve_compilations(
        [cm.exception, cm.exception.artifact]
    )
    self.assertGreaterEqual(resolved_count, 1)

  def test_concurrent_fx_to_mlir_thread_safety(self):
    artifacts = []
    inputs = [
        torch.randn(32 * (i + 1), dtype=torch.bfloat16, device="tpu")
        for i in range(5)
    ]
    for x in inputs:
      graph = torch.fx.symbolic_trace(SimpleModel())
      artifacts.append(
          tpu_compiler_mod.StaticCompiler(async_compile=True)(graph, [x])
      )
    for artifact in artifacts:
      artifact.resolve()
    for artifact, x in zip(artifacts, inputs, strict=True):
      res = artifact([x])
      self.assertEqual(res.shape, x.shape)

  def test_concurrent_tpu_work_corrupting_traversal(self):
    def fn(value):
      return torch.sin(value + 1)

    # Build the FX graph before starting asynchronous compilation.
    graph = make_fx(fn)(torch.ones(1024))
    value = torch.ones(1024, device="tpu")

    original = tpu_compiler_mod.tpu_torch_compile.traverse_and_compile
    worker_ready = threading.Event()
    start_traversal = threading.Event()

    def delayed_traversal(*args, **kwargs):
      worker_ready.set()
      start_traversal.wait()
      return original(*args, **kwargs)

    with mock.patch.object(
        tpu_compiler_mod.tpu_torch_compile,
        "traverse_and_compile",
        side_effect=delayed_traversal,
    ):
      artifact = tpu_compiler_mod.StaticCompiler(async_compile=True)(
          graph, [value]
      )

      # Run TPU work after the worker captures its graph but before it
      # traverses it.
      worker_ready.wait()
      torch.arange(128, dtype=torch.int32, device="tpu").cpu()
      start_traversal.set()

      res = artifact([value])
      test_utils.assert_close(res.cpu(), torch.sin(value.cpu() + 1))

  def test_async_compile_debug_callback_waits_for_compile(self):
    barrier = CompileBarrier()
    self.addCleanup(barrier.release_all)
    model = SimpleModel().to("tpu")
    callback_called = threading.Event()
    captured_mlir_counts: list[int] = []

    def debug_cb(debug):
      captured_mlir_counts.append(len(debug.stablehlo_forward_text))
      callback_called.set()

    with mock.patch.object(
        tpu_compiler_mod.StaticCompiler,
        "_compile",
        barrier.intercept_compile(tpu_compiler_mod.StaticCompiler._compile),
    ):
      compiled = torch.compile(
          torch.no_grad()(model),
          backend="tpu",
          fullgraph=True,
          dynamic=False,
          options={"async_compile": True, "debug_callback": debug_cb},
      )
      with self.assertRaises(  # ASSERT_RAISES_OK=AsyncCompilationSubmitted warmup signal.
          tpu_compile.AsyncCompilationSubmitted
      ) as cm:
        compiled(torch.ones(4, 16, device="tpu"))

      self.assertTrue(barrier.wait_for_compile_start(0))
      self.assertFalse(
          callback_called.wait(timeout=0.5),
          "debug_callback fired before async compilation finished!",
      )
      barrier.release_all()
      cm.exception.artifact.resolve()
      self.assertTrue(callback_called.wait(timeout=5.0))
      self.assertEqual(captured_mlir_counts, [1])


if __name__ == "__main__":
  absltest.main()
