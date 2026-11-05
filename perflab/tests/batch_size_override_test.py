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

"""Tests that benchmark factories build at the workload of their context.

The harness passes the workload of each case to the factory as
`Context.workload`, e.g. to measure a benchmark at other batch sizes than that
of its default workload in a batch sweep. This builds one benchmark per model
loader under the FakeTensorMode of the CPU smoke test, which allocates no
storage, at its full-scope workload with another batch size, and checks its
inputs. The CPU smoke tests check every benchmark at its default workload,
which a factory that ignores `Context.workload` would also pass.
"""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from perflab import common
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import mode as mode_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab import workload as workload_lib
from tests import seed_test_utils

_IMPORT_FAILURES = discovery_lib.import_submodules(models)

# A batch size that no benchmark uses by default.
_BATCH_SIZE = 3

# One benchmark per model loader.
_CASES = (
    "bert_base_forward",  # hf_text
    "vit_base_forward",  # hf_vision
    "llama_1b_training",  # llama
    "gemma_3_270m_forward",  # gemma
    "qwen3_0_6b_training",  # qwen
    "starcoder2_3b_forward",  # hf_llm
    "whisper_base_forward",  # whisper
    "timm_resnet_50_forward",  # timm_models
    "conformer_forward",  # conformer
    "bert_base_uncased_forward_gen",  # gen_utils/load_model
)


class BatchSizeOverrideTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # `--fake_tensor_mode` is only valid on CPU.
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )
    self.enter_context(flagsaver.flagsaver(fake_tensor_mode=True))

  def test_models_import(self):
    self.assertEmpty(_IMPORT_FAILURES)

  @parameterized.parameters(*_CASES)
  def test_builds_at_another_batch_size(self, name: str):
    spec = registry_lib.REGISTRY[name]
    ctx = registry_lib.resolve_context(
        spec,
        context_lib.Context(
            target=target_lib.make_target(
                target_lib.Platform.CPU, dtype=spec.dtype
            ),
            run_scope=context_lib.RunScope.FULL,
        ),
        batch_size=_BATCH_SIZE,
    )
    self.assertEqual(ctx.require_workload().batch_size, _BATCH_SIZE)
    common.seed_rngs()
    with mode_lib.fake_tensor_context():
      benchmark_args = spec.factory(ctx)
    self.assertIsNone(
        workload_lib.check_built_inputs(ctx.require_workload(), benchmark_args)
    )


if __name__ == "__main__":
  absltest.main()
