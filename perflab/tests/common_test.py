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

"""Tests for perflab/common.py."""

from unittest import mock

from absl.testing import absltest
import torch
import torch.nn as nn
from perflab import common
from tests import seed_test_utils


class CommonTest(seed_test_utils.RepeatableTest):

  def test_prepare_inputs_canonicalizes_module_params_and_buffers_in_place(
      self,
  ):
    class _Module(nn.Module):

      def __init__(self):
        super().__init__()
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
        b = torch.arange(6, dtype=torch.int32).reshape(2, 3).t()
        self.weight = nn.Parameter(w)
        self.register_buffer("buf", b)

    model = _Module()
    self.assertFalse(model.weight.is_contiguous())
    self.assertFalse(model.buf.is_contiguous())
    weight_expected = model.weight.detach().clone()
    buf_expected = model.buf.detach().clone()
    weight_param_ref = model.weight
    buf_ref = model.buf

    common.prepare_inputs(model, ())

    self.assertIs(model.weight, weight_param_ref)
    self.assertIs(model.buf, buf_ref)
    self.assertTrue(model.weight.is_contiguous())
    self.assertTrue(model.buf.is_contiguous())
    self.assertEqual(model.weight.dtype, torch.float32)
    self.assertEqual(model.buf.dtype, torch.int32)
    self.assertTrue(torch.equal(model.weight.data, weight_expected))
    self.assertTrue(torch.equal(model.buf.data, buf_expected))

  def test_prepare_inputs_canonicalizes_nested_pytree_inputs_and_non_module_model(
      self,
  ):
    model_tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).t()
    in_a = torch.arange(8, dtype=torch.float16).reshape(2, 4).t()
    in_b = torch.arange(10, dtype=torch.int64).reshape(2, 5).t()
    in_bool = torch.tensor([[True, False, True], [False, True, False]]).t()
    self.assertFalse(model_tensor.is_contiguous())
    self.assertFalse(in_a.is_contiguous())
    self.assertFalse(in_b.is_contiguous())
    self.assertFalse(in_bool.is_contiguous())

    expected_model = model_tensor.clone()
    expected_a = in_a.clone()
    expected_b = in_b.clone()
    expected_bool = in_bool.clone()

    non_module_model = {"weights": [model_tensor], "name": "callable_model"}
    inputs = (
        [in_a],
        {"nested": (in_b, in_bool), "scalar": 42},
    )

    common.prepare_inputs(non_module_model, inputs)

    self.assertTrue(model_tensor.is_contiguous())
    self.assertTrue(in_a.is_contiguous())
    self.assertTrue(in_b.is_contiguous())
    self.assertTrue(in_bool.is_contiguous())
    self.assertEqual(model_tensor.dtype, torch.bfloat16)
    self.assertEqual(in_a.dtype, torch.float16)
    self.assertEqual(in_b.dtype, torch.int64)
    self.assertEqual(in_bool.dtype, torch.bool)
    self.assertTrue(torch.equal(model_tensor, expected_model))
    self.assertTrue(torch.equal(in_a, expected_a))
    self.assertTrue(torch.equal(in_b, expected_b))
    self.assertTrue(torch.equal(in_bool, expected_bool))

  def test_prepare_inputs_skips_meta_tensors(self):
    meta_param = nn.Parameter(
        torch.empty((3, 4), device="meta", dtype=torch.float32).t()
    )
    meta_input = torch.empty((2, 5), device="meta", dtype=torch.float32).t()
    cpu_input = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()

    model = nn.Module()
    model.register_parameter("meta_param", meta_param)

    common.prepare_inputs(model, (meta_input, cpu_input))

    self.assertTrue(meta_param.is_meta)
    self.assertFalse(meta_param.is_contiguous())
    self.assertTrue(meta_input.is_meta)
    self.assertFalse(meta_input.is_contiguous())
    self.assertTrue(cpu_input.is_contiguous())

  def test_prepare_inputs_deduplicates_shared_tensors(self):
    shared = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    model = nn.Module()
    param = nn.Parameter(shared)
    model.register_parameter("p1", param)

    with mock.patch.object(torch, "clone", wraps=torch.clone) as mock_clone:
      common.prepare_inputs(model, (param, [param], {"alias": param}))

    self.assertEqual(mock_clone.call_count, 1)
    self.assertTrue(param.is_contiguous())


class RunModeTest(seed_test_utils.RepeatableTest):

  def test_no_run_mode_is_a_prefix_of_another(self):
    values = [mode.value for mode in common.RunMode]
    for a in values:
      for b in values:
        if a != b:
          self.assertFalse(b.startswith(a), f"{a} is a prefix of {b}")


class IsOutOfMemoryTest(seed_test_utils.RepeatableTest):

  def test_torch_out_of_memory_error(self):
    self.assertTrue(
        common.is_out_of_memory(torch.OutOfMemoryError("CUDA out of memory."))
    )

  def test_resource_exhausted_message(self):
    self.assertTrue(
        common.is_out_of_memory(
            RuntimeError(
                "RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out"
                " of memory in memory space hbm."
            )
        )
    )

  def test_wrapped_error(self):
    try:
      try:
        raise RuntimeError("RESOURCE_EXHAUSTED: out of HBM")
      except RuntimeError as e:
        raise ValueError("backend compiler failed") from e
    except ValueError as wrapped:
      self.assertTrue(common.is_out_of_memory(wrapped))

  def test_other_error(self):
    self.assertFalse(common.is_out_of_memory(RuntimeError("shape mismatch")))


if __name__ == "__main__":
  absltest.main()
