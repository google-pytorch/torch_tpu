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

"""Tests for `harness/compile.py`.

Mainly checks for two things:

  - Validating config at construction: A bad scope must raise where the
    benchmark is declared.
  - compile_config.apply really compiles with the configured `dynamic` and
    `fullgraph` settings; if any field is dropped, it silently changes what is
    being measured.

The CPU tests run a real torch.compile (Inductor). The TPU backend cannot run
on CPU, so for TPU targets only the selected backend object is checked.
"""

from absl.testing import absltest
import torch
from torch import nn
from torch_tpu._internal import compile as torch_tpu_compile
from torch_tpu._internal.utils import test_utils
from perflab import compile as compile_lib
from perflab import target as target_lib
from tests import seed_test_utils


# TODO(wan): revisit DECISION (b/568592967): the test bodies only use matmuls,
# which Inductor lowers to extern ATen kernels, because the test sandbox has no
# C++ compiler for Inductor's generated CPU kernels.
_WEIGHT = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10


def _project(x: torch.Tensor) -> torch.Tensor:
  return torch.mm(x, _WEIGHT)


def _with_graph_break(x: torch.Tensor) -> torch.Tensor:
  y = torch.mm(x, _WEIGHT)
  torch._dynamo.graph_break()  # pylint: disable=protected-access
  return torch.mm(y, _WEIGHT.t())


def _unique_graphs() -> int:
  return torch._dynamo.utils.counters["stats"]["unique_graphs"]  # pylint: disable=protected-access


class CompileConfigTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()  # pylint: disable=protected-access
    torch._dynamo.utils.counters.clear()  # pylint: disable=protected-access
    self.cpu = target_lib.make_target(target_lib.Platform.CPU)

  def test_default_is_step_scope(self):
    cfg = compile_lib.CompileConfig()
    self.assertEqual(cfg.scope, compile_lib.Scope.STEP)
    self.assertIsNone(cfg.dynamic)
    self.assertFalse(cfg.fullgraph)

  def test_scope_enum_values(self):
    self.assertEqual(compile_lib.Scope.MODEL.value, "model")
    self.assertEqual(compile_lib.Scope.STEP.value, "step")

  def test_frozen(self):
    with self.assertRaises(Exception):  # ASSERT_RAISES_OK=benchmark
      compile_lib.CompileConfig().scope = compile_lib.Scope.STEP  # type: ignore[misc]

  def test_bad_scope_raises(self):
    with self.assertRaises(ValueError):  # ASSERT_RAISES_OK=benchmark
      compile_lib.Scope("custom")

  def test_apply_compiles_module_matching_eager(self):
    model = nn.Linear(4, 3)
    x = torch.randn(2, 4)

    compiled = compile_lib.CompileConfig(scope=compile_lib.Scope.MODEL).apply(
        model, self.cpu
    )

    test_utils.assert_close(compiled(x), model(x))
    self.assertEqual(_unique_graphs(), 1)

  def test_apply_compiles_step_callable_matching_eager(self):
    x = torch.randn(3, 5)

    compiled = compile_lib.CompileConfig(scope=compile_lib.Scope.STEP).apply(
        _project, self.cpu
    )

    test_utils.assert_close(compiled(x), _project(x))
    self.assertEqual(_unique_graphs(), 1)

  def test_fullgraph_rejects_graph_break(self):
    compiled = compile_lib.CompileConfig(fullgraph=True).apply(
        _with_graph_break, self.cpu
    )

    with self.assertRaises(torch._dynamo.exc.Unsupported):  # pylint: disable=protected-access  # ASSERT_RAISES_OK=benchmark
      compiled(torch.ones(2, 5))

  def test_default_allows_graph_break(self):
    compiled = compile_lib.CompileConfig().apply(_with_graph_break, self.cpu)

    x = torch.ones(2, 5)
    test_utils.assert_close(compiled(x), _with_graph_break(x))

  def test_dynamic_controls_recompiles_across_shapes(self):
    # dynamic=True traces once with symbolic shapes; dynamic=None
    # (automatic dynamic) recompiles once as dynamic after the first shape
    # change; dynamic=False recompiles for every new shape.
    for dynamic, expected_graphs in ((True, 1), (None, 2), (False, 3)):
      with self.subTest(dynamic=dynamic):
        torch._dynamo.reset()  # pylint: disable=protected-access
        torch._dynamo.utils.counters.clear()  # pylint: disable=protected-access
        compiled = compile_lib.CompileConfig(dynamic=dynamic).apply(
            _project, self.cpu
        )
        for rows in (2, 3, 4):
          x = torch.randn(rows, 5)
          test_utils.assert_close(compiled(x), _project(x))
        self.assertEqual(_unique_graphs(), expected_graphs)

  def test_cpu_selects_inductor_and_configures_it(self):
    with torch._inductor.config.patch(compile_threads=8):  # pylint: disable=protected-access
      backend = compile_lib.CompileConfig().select_backend(self.cpu)
      self.assertEqual(torch._inductor.config.compile_threads, 1)  # pylint: disable=protected-access
    self.assertEqual(backend, "inductor")

  def test_cuda_selects_inductor(self):
    target = target_lib.make_target(target_lib.Platform.B200_1)
    self.assertEqual(
        compile_lib.CompileConfig().select_backend(target), "inductor"
    )

  def test_tpu_backend_dynamism_follows_dynamic(self):
    target = target_lib.make_target(target_lib.Platform.V5E_1X1)
    # TODO(wan): revisit DECISION (b/568592967): TpuBackend only exposes
    # dynamism as a private attribute and cannot compile on CPU, so the real
    # backend object's state is checked instead of running a TPU compile.
    for dynamic, expected_dynamism in (
        (None, True),
        (True, True),
        (False, False),
    ):
      with self.subTest(dynamic=dynamic):
        backend = compile_lib.CompileConfig(dynamic=dynamic).select_backend(
            target
        )
        self.assertIsInstance(backend, torch_tpu_compile.TpuBackend)
        self.assertIs(backend._dynamism, expected_dynamism)  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
