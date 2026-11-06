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

"""Tests for perflab/model_helpers.py."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import torch
from perflab import context as context_lib
from perflab import model_helpers
from perflab import target as target_lib
from tests import seed_test_utils


def _context(run_scope: context_lib.RunScope) -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(target_lib.Platform.CPU),
      run_scope=run_scope,
  )


class IsPresubmitSmokeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # `--fake_tensor_mode` is only valid on CPU.
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )

  def test_presubmit_with_fake_tensor_mode(self):
    with flagsaver.flagsaver(fake_tensor_mode=True):
      self.assertTrue(
          model_helpers.is_presubmit_smoke(
              _context(context_lib.RunScope.PRESUBMIT)
          )
      )

  def test_presubmit_without_fake_tensor_mode(self):
    with flagsaver.flagsaver(fake_tensor_mode=False):
      self.assertFalse(
          model_helpers.is_presubmit_smoke(
              _context(context_lib.RunScope.PRESUBMIT)
          )
      )

  def test_full_scope(self):
    for fake_tensor_mode in (False, True):
      with self.subTest(fake_tensor_mode=fake_tensor_mode):
        with flagsaver.flagsaver(fake_tensor_mode=fake_tensor_mode):
          self.assertFalse(
              model_helpers.is_presubmit_smoke(
                  _context(context_lib.RunScope.FULL)
              )
          )


class _FakeModuleSpec:
  """Fake transformers module spec of a small model."""

  def __init__(self):
    self.factory_dtypes = []

  def module_factory(self, dtype: torch.dtype | None = None) -> torch.nn.Module:
    self.factory_dtypes.append(dtype)
    return torch.nn.Linear(4, 4, dtype=dtype)

  def sample_inputs_factory(self, shape, device_str):
    del device_str  # Unused.
    return None, {"input_ids": torch.zeros(shape, dtype=torch.long)}


class LoadHfLlmTest(seed_test_utils.RepeatableTest):

  def test_constructs_model_in_benchmark_dtype(self):
    # The registry constructs the model in the benchmark dtype. Constructing it
    # in float32 and casting it afterwards needs twice the device memory of
    # the model.
    module_spec = _FakeModuleSpec()
    registry = mock.Mock()
    registry.get_module_spec.return_value = module_spec

    with mock.patch.object(
        model_helpers, "get_module_registry", return_value=registry
    ):
      model, _ = model_helpers.load_hf_llm(
          "fake/model", dtype=torch.bfloat16, device_str="cpu", seq_len=8
      )

    self.assertEqual(module_spec.factory_dtypes, [torch.bfloat16])
    self.assertEqual(model.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
  absltest.main()
