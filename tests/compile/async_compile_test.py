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
import concurrent.futures
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import _backend as tpu_backend_mod
from torch_tpu._internal.compile import compiler as tpu_compiler_mod
from torch_tpu._internal.utils import test_utils


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

  pass


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
    orig_resolve = artifact._resolve

    def strict_resolve():
      if not self.resolution_allowed:
        raise PrematureResolutionError(
            f"Artifact {artifact} was resolved prematurely! "
            "A PyTorch wrapper or Dynamo guard accessed an attribute that"
            " triggered _resolve()."
        )
      return orig_resolve()

    artifact._resolve = strict_resolve
    return artifact

  def allow_resolution(self):
    self.resolution_allowed = True


def _reset() -> None:
  """Resets torch compiler, dynamo, and AOTAutograd TPU cache."""
  torch.compiler.reset()
  if hasattr(torch, "tpu") and hasattr(torch.tpu, "_clear_cache"):
    torch.tpu._clear_cache()
  torch._dynamo.reset()
  try:
    from torch._functorch._aot_autograd import autograd_cache  # pylint: disable=g-import-not-at-top

    autograd_cache.AOTAutogradCache.clear()
  except (ImportError, AttributeError):
    pass


class AsyncCompileTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    tt_testing.reset_eager_state()
    _reset()

  def test_async_compile_basic(self):
    _reset()
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
    _reset()
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
          options={"experimental_async_compile": True},
      )

      # 1. Warmup Shape 1
      with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm1:
        compiled(*shape1)
      art1 = cm1.exception.artifact
      self.assertIsNotNone(art1)
      self.assertFalse(art1.is_resolved)

      # Verify background compilation for Shape 1 has actually started and is waiting
      self.assertTrue(barrier.wait_for_compile_start(0))

      # 2. Warmup Shape 2 while Shape 1 is STILL blocked in compilation
      with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm2:
        compiled(*shape2)
      art2 = cm2.exception.artifact
      self.assertIsNotNone(art2)

      # Shape 2 Dynamo/AOTAutograd tracing has completed on main thread, but neither is resolved
      self.assertFalse(
          art1.is_resolved,
          "Shape 1 resolved prematurely during Shape 2 tracing!",
      )
      self.assertFalse(art2.is_resolved, "Shape 2 resolved prematurely!")

      # 3. Release Shape 1 and Shape 2 compilations
      barrier.release_all()

      # 4. Resolve barrier
      resolved = tpu_compile.resolve_compilations([art1, art2])
      self.assertEqual(resolved, 2)
      self.assertTrue(art1.is_resolved)
      self.assertTrue(art2.is_resolved)

      # 5. Verify execution on TPU produces correct results
      out1 = compiled(*shape1)
      out2 = compiled(*shape2)
      test_utils.assert_close(out1.cpu(), (shape1[0] + 1.0).cpu())
      test_utils.assert_close(out2.cpu(), (shape2[0] + 1.0).cpu())

  def test_async_compile_zero_premature_resolution_canary(self):
    """Ensures no PyTorch or backend wrapper triggers premature artifact resolution."""
    _reset()
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
          options={"experimental_async_compile": True},
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
    _reset()
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

  def test_async_compile_complex_pytree_outputs(self):
    """Tests async compilation with multi-tensor and dictionary pytree outputs."""
    _reset()
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
    _reset()
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
    _reset()
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
          options={"experimental_async_compile": True},
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
    _reset()
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

  def test_async_compile_multithreaded_safety(self):
    """Verifies thread-safety when async_compile is called concurrently."""
    _reset()
    models = [SimpleModel().to("tpu"), AnotherModel().to("tpu")]

    def run_async(m, val):
      inputs = [(torch.randn(2 * i, 4, device="tpu"),) for i in range(1, 4)]
      comp = tpu_compile.async_compile(m, inputs)
      res = comp(torch.ones(4, 4, device="tpu"))
      return res.cpu()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      f1 = executor.submit(run_async, models[0], 1.0)
      f2 = executor.submit(run_async, models[1], 2.0)
      res1 = f1.result(timeout=20.0)
      res2 = f2.result(timeout=20.0)

    test_utils.assert_close(res1, torch.ones(4, 4) + 1.0)
    test_utils.assert_close(res2, torch.ones(4, 4) * 2.0)

  def test_async_compile_no_grad_option(self):
    _reset()
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
    _reset()
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
    _reset()
    model = AnotherModel().to("tpu")
    compiled_model = torch.compile(
        torch.no_grad()(model),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
        options={"experimental_async_compile": True},
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
    _reset()
    model = AnotherModel().to("tpu")
    compiled_model = torch.compile(
        torch.no_grad()(model),
        backend="tpu",
        fullgraph=True,
        dynamic=False,
        options={"experimental_async_compile": True},
    )

    x = torch.ones(3, 7, device="tpu")
    with self.assertRaises(tpu_compile.AsyncCompilationSubmitted) as cm:
      compiled_model(x)

    resolved_count = tpu_compile.resolve_compilations(
        [cm.exception, cm.exception.artifact]
    )
    self.assertGreaterEqual(resolved_count, 1)

  def test_async_aot_cache_saves_context(self):
    with tpu_backend_mod._async_aot_cache_saves():
      pass
    self.assertIsNotNone(tpu_backend_mod._cache_save_executor)

  def test_concurrent_fx_to_mlir_thread_safety(self):
    _reset()
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


if __name__ == "__main__":
  absltest.main()
