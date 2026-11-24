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

"""CPU unit tests for DynamicCompiler and DynamicTpuCompiledExecutable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch._dynamo.backends import common as dynamo_common
from torch_tpu._internal.compile import compiler
from torch_tpu._internal.compile.dynamic import dynamic_compiler
from torch_tpu._internal.compile.dynamic import dynamic_executable
from torch_tpu._internal.compile.dynamic import sym_bounds_manager
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils


class _FakeCompiledArtifact(compiler.CompiledArtifact):

  def __call__(
      self,
      inputs: Sequence[Any],
      output_shapes: Any = None,
  ) -> tuple[torch.Tensor]:
    del output_shapes
    scalar_tensor, weight = inputs
    return (weight + scalar_tensor.to(weight.dtype),)

  def __reduce__(self):
    return (_FakeCompiledArtifact, ())


class _FakeStaticCompiler(compiler.StaticCompiler):

  def __init__(self):
    super().__init__(async_compile=False)

  def __call__(
      self,
      graph_module: torch.fx.GraphModule,
      example_inputs: Sequence[Any],
      **kwargs: Any,
  ) -> compiler.CompiledArtifact:
    del graph_module, example_inputs, kwargs
    return _FakeCompiledArtifact()


class DynamicCompilerTest(seed_test_utils.RepeatableTest):

  def test_zero_element_dynamic_input_check_is_guard_free(self):
    seen: dict[str, Any] = {}

    def fw_compiler(graph_module, example_inputs):
      shape_env = next(
          x.node.shape_env
          for x in example_inputs
          if isinstance(x, torch.SymInt)
      )
      guards_before = set(shape_env.guards)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python backend error.
          NotImplementedError,
          r"Zero-element dynamic input tensor.*dynamic=False",
      ):
        dynamic_compiler.DynamicCompiler()(graph_module, list(example_inputs))
      seen["guards_before"] = guards_before
      seen["guards_after"] = set(shape_env.guards)
      return graph_module

    @torch.compile(backend=dynamo_common.aot_autograd(fw_compiler=fw_compiler))
    def fn(x):
      return x * 2 + 1

    x = torch.zeros((0, 6), dtype=torch.int64)
    # Leave upper bound unset so SymBoundsManager.add_bound_checks would add
    # a guard onto shape_env.guards if reached before validation.
    torch._dynamo.mark_dynamic(x, 1)
    fn(x)

    self.assertEqual(seen["guards_after"], seen["guards_before"])

  def test_validate_dynamic_example_inputs(self):
    seen: dict[str, bool] = {}

    def fw_compiler(graph_module, example_inputs):
      traced = list(example_inputs)
      sym = next(x for x in traced if isinstance(x, torch.SymInt))
      dyn_nonzero = next(x for x in traced if isinstance(x, torch.Tensor))
      static_zero = torch.zeros((0, 4), dtype=torch.float32)

      # Static zero-element tensors, scalars, None, and non-zero dynamic tensors
      # pass validation without raising.
      dynamic_executable.validate_dynamic_example_inputs(
          [None, sym, static_zero, 2.5, dyn_nonzero]
      )

      fake_mode = torch._guards.detect_fake_mode(traced)
      with fake_mode:
        dyn_zero = torch.empty((0, sym), dtype=torch.float32)
      with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Python backend error.
          NotImplementedError,
          r"Zero-element dynamic input tensor at position 2 \(shape \[0, s",
      ):
        dynamic_executable.validate_dynamic_example_inputs(
            [static_zero, dyn_nonzero, dyn_zero]
        )
      seen["ran"] = True
      return graph_module

    @torch.compile(
        backend=dynamo_common.aot_autograd(fw_compiler=fw_compiler),
        dynamic=True,
    )
    def fn(x):
      return x + 1

    fn(torch.ones((4, 8)))
    self.assertTrue(seen["ran"])

  def _make_mixed_executables(self):
    out: dict[str, Any] = {}

    def fw_compiler(graph_module, example_inputs):
      traced = list(example_inputs)
      sym = next(x for x in traced if isinstance(x, torch.SymInt))
      dyn_x = next(x for x in traced if isinstance(x, torch.Tensor))
      shape_env = sym.node.shape_env
      bounds = sym_bounds_manager.SymBoundsManager(
          traced
      ).get_fallback_symbol_bounds()
      mixed_inputs = [None, sym, dyn_x, 0.5, torch.ones(3)]
      bounded_inputs, input_bounds = (
          dynamic_compiler._create_bounded_model_inputs(mixed_inputs, bounds)  # pylint: disable=protected-access
      )
      spec = dynamic_executable.BucketCompilationSpec(
          symbol_bounds=bounds,
          graph_module=graph_module,
          bounded_model_inputs=bounded_inputs,
          model_input_bounds=input_bounds,
      )
      guards_before = len(shape_env.guards)
      for key, use_in_graph in (
          ("in_graph", True),
          ("pad_subgraph", False),
      ):
        exe = dynamic_executable.DynamicTpuCompiledExecutable(
            static_compiler=_FakeStaticCompiler(),
            example_inputs=mixed_inputs,
            bucket_specs=[spec],
            output_sym_shapes=[],
            is_fwd=True,
            view_arg_indices=set(),
            use_pad_real_to_static=use_in_graph,
        )
        exe._backend_device = torch.device("cpu")  # pylint: disable=protected-access
        out[key] = exe
      out["guards_added"] = len(shape_env.guards) - guards_before
      return graph_module

    @torch.compile(
        backend=dynamo_common.aot_autograd(fw_compiler=fw_compiler),
        dynamic=True,
    )
    def fn(x):
      return x + 1

    fn(torch.ones((4, 8)))
    return out["in_graph"], out["pad_subgraph"], out["guards_added"]

  def test_dynamic_executable_input_packing_plan(self):
    exe_in_graph, exe_pad_subgraph, guards_added = (
        self._make_mixed_executables()
    )
    self.assertEqual(guards_added, 0)
    self.assertEqual(exe_in_graph._dynamic_target_indices, {1})  # pylint: disable=protected-access
    self.assertEqual(exe_pad_subgraph._dynamic_target_indices, {1})  # pylint: disable=protected-access
    self.assertEqual(exe_in_graph._static_tensor_map, [(2, 1), (4, 3)])  # pylint: disable=protected-access
    self.assertEmpty(exe_in_graph._dynamic_tensor_map)  # pylint: disable=protected-access
    self.assertEqual(exe_pad_subgraph._static_tensor_map, [(4, 3)])  # pylint: disable=protected-access
    self.assertEqual(exe_pad_subgraph._dynamic_tensor_map, [(0, 1)])  # pylint: disable=protected-access

    meta_dyn = torch.empty((4, 8), device="meta")
    stat_y = torch.ones(3)
    in_graph_inputs = exe_in_graph._get_model_inputs(  # pylint: disable=protected-access
        [None, 4, meta_dyn, 0.5, stat_y]
    )
    self.assertIs(in_graph_inputs[1], meta_dyn)
    self.assertIs(in_graph_inputs[3], stat_y)

    padded_dyn = torch.empty((16, 8), device="meta")
    subgraph_inputs = exe_pad_subgraph._get_model_inputs(  # pylint: disable=protected-access
        [None, 4, meta_dyn, 0.5, stat_y], [padded_dyn]
    )
    self.assertIs(subgraph_inputs[1], padded_dyn)

  @parameterized.named_parameters(
      ("in_graph_pad", True),
      ("pad_subgraph", False),
  )
  def test_dynamic_executable_symint_only_call(
      self, use_pad_real_to_static: bool
  ):
    holder: dict[str, Any] = {}

    def fw_compiler(graph_module, example_inputs):
      traced = list(example_inputs)
      sym = next(x for x in traced if isinstance(x, torch.SymInt))
      bounds = sym_bounds_manager.SymBoundsManager(
          traced
      ).get_fallback_symbol_bounds()
      sym_only_inputs = [sym, torch.ones(3)]
      sym_bounded, sym_input_bounds = (
          dynamic_compiler._create_bounded_model_inputs(sym_only_inputs, bounds)  # pylint: disable=protected-access
      )
      sym_spec = dynamic_executable.BucketCompilationSpec(
          symbol_bounds=bounds,
          graph_module=graph_module,
          bounded_model_inputs=sym_bounded,
          model_input_bounds=sym_input_bounds,
      )
      sym_exe = dynamic_executable.DynamicTpuCompiledExecutable(
          static_compiler=_FakeStaticCompiler(),
          example_inputs=sym_only_inputs,
          bucket_specs=[sym_spec],
          output_sym_shapes=[],
          is_fwd=True,
          view_arg_indices=set(),
          use_pad_real_to_static=use_pad_real_to_static,
      )
      sym_exe._backend_device = torch.device("cpu")  # pylint: disable=protected-access
      holder["sym_exe"] = sym_exe
      return graph_module

    @torch.compile(
        backend=dynamo_common.aot_autograd(fw_compiler=fw_compiler),
        dynamic=True,
    )
    def fn(x):
      return x + 1

    fn(torch.ones((4, 8)))
    (got_out,) = holder["sym_exe"](5, torch.ones(3))
    utils.assert_close(got_out, torch.full((3,), 6.0))

  def test_dynamic_executable_runtime_validation_errors(self):
    exe_in_graph, _, _ = self._make_mixed_executables()
    stat_y = torch.ones(3)
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Internal runtime check.
        ValueError,
        "CPU tensors are not supported in dynamic shapes compilation",
    ):
      exe_in_graph(None, 4, torch.ones((4, 8), device="cpu"), 0.5, stat_y)

    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Internal runtime check.
        NotImplementedError,
        r"Zero-element dynamic input tensor at position 2 \(shape \[0, 8\]\)",
    ):
      exe_in_graph(None, 4, torch.empty((0, 8), device="meta"), 0.5, stat_y)


if __name__ == "__main__":
  absltest.main()
