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

"""Unit tests for experimental asynchronous compilation error handling."""

from collections.abc import Callable
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd import autograd_cache
from torch._inductor.codecache import FxGraphCache
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import clear_caches
from torch.compiler._cache import CacheArtifactManager
from torch_tpu._internal import compile as tpu_compile
from torch_tpu._internal import testing as tt_testing
from torch_tpu._internal.compile import compiler as tpu_compiler_mod
from tests import seed_test_utils


class SimpleModel(nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + 1.0


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

  def _submit_async_compile(
      self, fn: Callable[..., Any], *args: Any, **kwargs: Any
  ):
    with self.assertRaises(  # ASSERT_RAISES_OK=AsyncCompilationSubmitted is normal control flow to return compilation future
        tpu_compile.AsyncCompilationSubmitted
    ) as ctx:
      fn(*args, **kwargs)
    return ctx.exception

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

      exc = self._submit_async_compile(compiled, torch.ones(4, 4, device="tpu"))
      art = exc.artifact
      self.assertIsNotNone(art)

      with self.assertRaisesRegex(
          RuntimeError, "Simulated XLA lowering failure"
      ):
        art.resolve()


if __name__ == "__main__":
  absltest.main()
