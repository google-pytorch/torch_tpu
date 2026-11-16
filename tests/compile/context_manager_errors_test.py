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

"""Tests for errors raised by TorchTPU context managers."""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal import compiler_options
from torch_tpu._internal import execution_mode
from torch_tpu._internal import precision
from torch_tpu._internal import profiler
from torch_tpu._internal.device_utils import annotations
from torch_tpu._internal.export import export
from torch_tpu._internal.profiler import named_scope
from tests import seed_test_utils

# See the assertion text in torch/_dynamo/symbolic_convert.py.
_ASSERTION_ERROR_REGEX = r"Attempt to trace forbidden callable"


class ContextManagerErrorsTest(seed_test_utils.RepeatableTest):
  """Tests that banned TorchTPU context managers fail fast under Dynamo."""

  def setUp(self):
    super().setUp()
    # Drop cached compilations so each test observes a fresh trace.
    torch._dynamo.reset()

  def test_set_eager_mode_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_layout_context_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with annotations.LayoutContext(
          annotations.TpuLayout(minor_to_major=[1, 0])
      ):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_custom_compiler_options_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with compiler_options.custom_compiler_options(
          {"xla_optimization_level": "O2"}
      ):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_enable_tracebacks_raises_in_torch_compile_region(self):

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with export.enable_tracebacks(False):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_profile_raises_in_torch_compile_region(self):
    handler = profiler.xprof_trace_handler(self.create_tempdir().full_path)

    @torch.compile(backend="tpu")
    def model(x: torch.Tensor) -> torch.Tensor:
      with profiler.profile(
          activities=[profiler.ProfilerActivity.CPU],
          on_trace_ready=handler,
      ):
        return x.abs()

    with self.assertRaisesRegex(AssertionError, _ASSERTION_ERROR_REGEX):
      model(torch.randn(2, 2))

  def test_precision_raises_in_torch_compile_region_when_fullgraph(self):
    # With `fullgraph=True`, a graph break becomes a hard compile error instead
    # of a silent eager fallback.
    @torch.compile(backend="tpu", fullgraph=True)
    def model(x: torch.Tensor) -> torch.Tensor:
      with precision.precision(precision.Precision.HIGHEST):
        return x.abs()

    # TODO(b/568859422): remove the expected exception after `precision` context
    # manager becomes traceable.
    with self.assertRaisesRegex(
        torch._dynamo.exc.Unsupported,
        r"`precision` context manager is not yet traceable",
    ):
      model(torch.randn(2, 2))


class NamedScopeErrorsTest(seed_test_utils.RepeatableTest):

  def test_rejects_empty_name(self):
    with self.assertRaisesRegex(
        ValueError, r"named_scope expects a non-empty string, got ''"
    ):
      with named_scope.named_scope(""):
        pass

  def test_rejects_at_sign(self):
    # XLA truncates HLO op metadata names at the first "@".
    with self.assertRaisesRegex(ValueError, r"must not contain '@'"):
      with named_scope.named_scope("layer@1"):
        pass

  def test_rejects_colon(self):
    # XProf splits `<op_name>:<op_type>` at the first ":".
    with self.assertRaisesRegex(ValueError, r"must not contain ':'"):
      with named_scope.named_scope("step:1"):
        pass

  @parameterized.named_parameters(
      ("leading_slash", "/a"),
      ("trailing_slash", "a/"),
      ("consecutive_slashes", "a//b"),
  )
  def test_rejects_empty_segments(self, name: str):
    with self.assertRaisesRegex(ValueError, r"empty '/'-separated"):
      with named_scope.named_scope(name):
        pass

  def test_rejects_invalid_name_in_compiled_code(self):
    def model(x: torch.Tensor) -> torch.Tensor:
      with named_scope.named_scope(""):
        return x.abs()

    # Without `fullgraph`, Dynamo graph-breaks on the raised exception and the
    # frame re-raises it when it runs eagerly.
    with self.assertRaisesRegex(
        ValueError, r"named_scope expects a non-empty string, got ''"
    ):
      torch.compile(model, backend="eager")(torch.randn(2, 2))

  def test_rejects_invalid_name_in_compiled_code_when_fullgraph(self):
    def model(x: torch.Tensor) -> torch.Tensor:
      with named_scope.named_scope(""):
        return x.abs()

    # With `fullgraph=True`, Dynamo surfaces the exception it observed while
    # tracing (as `Unsupported` carrying the `ValueError`) instead of silently
    # compiling an unscoped graph.
    with self.assertRaisesRegex(
        (ValueError, torch._dynamo.exc.Unsupported),
        r"named_scope expects a non-empty string, got ''",
    ):
      torch.compile(model, backend="eager", fullgraph=True)(torch.randn(2, 2))


if __name__ == "__main__":
  absltest.main()
