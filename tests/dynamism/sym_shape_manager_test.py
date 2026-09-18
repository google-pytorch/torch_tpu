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

"""Unit tests for SymShapeManager."""

from __future__ import annotations
from absl.testing import absltest
import torch
from torch._dynamo.backends.common import aot_autograd
from torch_tpu._internal.compile.dynamic import sym_bounds_manager
from torch_tpu._internal.compile.dynamic import sym_shape_manager
from torch_tpu._internal.compile.dynamic import sym_utils
from tests import seed_test_utils


class SymShapeManagerTest(seed_test_utils.RepeatableTest):

  def test_has_dynamic_shape(self):
    captured_gm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_gm
      captured_gm = graph_module
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, y):
      return x + 1, y + 1

    t1 = torch.ones(4)
    t2 = torch.ones(4)
    torch._dynamo.mark_dynamic(t1, 0, min=2, max=8)

    f(t1, t2)

    self.assertIsNotNone(captured_gm)
    placeholders = list(
        captured_gm.graph.find_nodes(op="placeholder", sort=True)
    )
    # Dynamic placeholder vs static placeholder
    self.assertTrue(sym_utils.has_dynamic_shape(placeholders[1]))
    self.assertFalse(sym_utils.has_dynamic_shape(placeholders[2]))

  def test_symbol_manager_extraction(self):
    captured_sm = None
    captured_bm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_sm, captured_bm
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
      )
      captured_bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x):
      return x + 1

    t = torch.ones(4)
    torch._dynamo.mark_dynamic(t, 0, min=2, max=8)

    f(t)

    self.assertIsNotNone(captured_sm)
    self.assertIsNotNone(captured_bm)
    self.assertEqual(captured_sm.get_num_dynamic_dims(1), 1)
    self.assertEqual(captured_sm.input_tensors_metadata[1].static_shape, [-1])
    self.assertEqual(captured_bm.get_bounded_shape(1), [8])

  def test_symbol_manager_static_shape(self):
    captured_sm = None
    captured_bm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_sm, captured_bm
      captured_sm = sym_shape_manager.SymShapeManager(
          graph_module, example_inputs
      )
      captured_bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler))
    def f(x):
      return x + 1

    t = torch.ones(2, 3)

    f(t)

    self.assertIsNotNone(captured_sm)
    self.assertIsNotNone(captured_bm)
    self.assertEqual(captured_sm.get_num_dynamic_dims(0), 0)
    self.assertEqual(captured_sm.input_tensors_metadata[0].static_shape, [2, 3])
    self.assertEqual(captured_bm.get_bounded_shape(0), [2, 3])

  def test_validate_bounds_with_multi_symbol_sum_guard(self):
    captured_bm = None

    def fw_compiler(graph_module, example_inputs):
      nonlocal captured_bm
      captured_bm = sym_bounds_manager.SymBoundsManager(example_inputs)
      return graph_module

    @torch.compile(backend=aot_autograd(fw_compiler=fw_compiler), dynamic=True)
    def f(x, y):
      assert x.shape[0] + y.shape[0] == 64
      return x * 2, y * 3

    t1 = torch.ones(20)
    t2 = torch.ones(44)
    torch._dynamo.mark_dynamic(t1, 0)
    torch._dynamo.mark_dynamic(t2, 0)

    f(t1, t2)

    self.assertIsNotNone(captured_bm)
    # validate_bounds() should not raise ConstraintViolationError on multi-symbol sum guard
    captured_bm.validate_bounds()


if __name__ == "__main__":
  absltest.main()
