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

"""Tests that presubmit scope runs large models only in the CPU smoke.

Large models run in presubmit scope only under the FakeTensorMode CPU smoke
(`--fake_tensor_mode`), which allocates no storage. Without it they would build
their full weights on the host, so their `PresubmitPolicy.CPU_SMOKE_ONLY` keeps
them out of every other presubmit-scope run.
"""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.models import gemma
from perflab.models import gpt
from perflab.models import qwen
from tests import seed_test_utils

# The models that are too large to build for real in presubmit scope, with the
# module and name of the HuggingFace loader that each factory builds it with.
_LARGE_MODEL_CASES = (
    {
        "testcase_name": "gemma_4_31b_6_layers_training",
        "factory": gemma.gemma_4_31b_6_layers_training,
        "module": gemma,
        "loader_name": "_load_hf_gemma",
    },
    {
        "testcase_name": "gemma_4_e2b_training",
        "factory": gemma.gemma_4_e2b_training,
        "module": gemma,
        "loader_name": "_load_hf_gemma",
    },
    {
        "testcase_name": "gemma_4_26b_a4b_6_layers_training",
        "factory": gemma.gemma_4_26b_a4b_6_layers_training,
        "module": gemma,
        "loader_name": "_load_hf_gemma",
    },
    {
        "testcase_name": "gpt_oss_20b_forward",
        "factory": gpt.gpt_oss_20b_forward,
        "module": gpt,
        "loader_name": "_load_hf_gpt",
    },
    {
        "testcase_name": "gpt_oss_20b_decode",
        "factory": gpt.gpt_oss_20b_decode,
        "module": gpt,
        "loader_name": "_load_hf_gpt",
    },
    {
        "testcase_name": "gpt_oss_20b_4_layers_training",
        "factory": gpt.gpt_oss_20b_4_layers_training,
        "module": gpt,
        "loader_name": "_load_hf_gpt",
    },
    {
        "testcase_name": "gpt_oss_120b_4_layers_forward",
        "factory": gpt.gpt_oss_120b_4_layers_forward,
        "module": gpt,
        "loader_name": "_load_hf_gpt",
    },
    {
        "testcase_name": "gpt_oss_120b_4_layers_decode",
        "factory": gpt.gpt_oss_120b_4_layers_decode,
        "module": gpt,
        "loader_name": "_load_hf_gpt",
    },
    {
        "testcase_name": "gpt_oss_120b_2_layers_training",
        "factory": gpt.gpt_oss_120b_2_layers_training,
        "module": gpt,
        "loader_name": "_load_hf_gpt",
    },
    {
        "testcase_name": "qwen3_5_397b_4_layers_forward",
        "factory": qwen.qwen3_5_397b_4_layers_forward,
        "module": qwen,
        "loader_name": "_load_hf_qwen",
    },
    {
        "testcase_name": "qwen3_5_397b_4_layers_decode",
        "factory": qwen.qwen3_5_397b_4_layers_decode,
        "module": qwen,
        "loader_name": "_load_hf_qwen",
    },
)


def _cpu_presubmit_context() -> context_lib.Context:
  return context_lib.Context(
      target=target_lib.make_target(target_lib.Platform.CPU),
      run_scope=context_lib.RunScope.PRESUBMIT,
  )


class LargeModelPresubmitScopeTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )
    self.enter_context(
        flagsaver.flagsaver(run_scope=context_lib.RunScope.PRESUBMIT)
    )

  @parameterized.named_parameters(*_LARGE_MODEL_CASES)
  def test_runs_in_presubmit_scope_only_in_cpu_smoke(
      self, factory, module, loader_name
  ):
    del module, loader_name  # Unused.
    spec = registry_lib.REGISTRY[factory.__name__]
    self.assertIs(spec.presubmit, registry_lib.PresubmitPolicy.CPU_SMOKE_ONLY)
    self.assertIn("uses too much memory", spec.presubmit_reason)

  @parameterized.named_parameters(*_LARGE_MODEL_CASES)
  @flagsaver.flagsaver(fake_tensor_mode=True)
  def test_cpu_presubmit_with_fake_tensor_mode_builds_model(
      self, factory, module, loader_name
  ):
    with mock.patch.object(
        module, loader_name, return_value=(torch.nn.Linear(1, 1), {})
    ) as load:
      factory(
          registry_lib.resolve_context(
              registry_lib.REGISTRY[factory.__name__], _cpu_presubmit_context()
          )
      )
    load.assert_called_once()


if __name__ == "__main__":
  absltest.main()
