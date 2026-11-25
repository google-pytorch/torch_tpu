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

"""Tests the FLOPs that perflab/flops.py counts for generated benchmarks."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
import torch
from torch.utils import flop_counter
from perflab import context as context_lib
from perflab import discovery as discovery_lib
from perflab import flops as flops_lib
from perflab import models
from perflab import registry as registry_lib
from perflab import steps
from perflab import target as target_lib
from tests import seed_test_utils

# Registers the generated benchmarks and the steppers they resolve.
discovery_lib.import_submodules(models)
discovery_lib.import_submodules(steps)

# The CPU attention ops that `torch.utils.flop_counter` has formulas for only
# from torch 2.15 on.
_CPU_SDPA_OPS = (
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,  # pylint: disable=protected-access
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward,  # pylint: disable=protected-access
)

_MODERNBERT_FORWARD = "time_anchor_modernbert_32m_forward_gen"
# The `model_flops_per_step` of the nightly runs in google3, at the full run
# scope.
_MODERNBERT_FORWARD_FLOPS = 115.76e9


class GenBenchmarkFlopsTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    # `--fake_tensor_mode` reads the model configs in-tree rather than from
    # the weights cache, which tests cannot reach, and is only valid on CPU.
    self.enter_context(
        mock.patch.dict(os.environ, {target_lib.PLATFORM_ENV_VAR: "cpu"})
    )
    self.enter_context(flagsaver.flagsaver(fake_tensor_mode=True))

  def _count_modernbert_forward(self) -> int:
    spec = registry_lib.REGISTRY[_MODERNBERT_FORWARD]
    run_scope = context_lib.RunScope.FULL
    workload = registry_lib.resolve_workload(
        spec,
        context_lib.Context(
            target=target_lib.make_target(target_lib.Platform.CPU),
            run_scope=run_scope,
        ),
    )
    flop_count = flops_lib.count_step_flops(spec, run_scope, workload=workload)
    self.assertIsNotNone(flop_count)
    return flop_count.total

  def test_counts_attention_of_modernbert(self):
    self.assertAlmostEqual(
        self._count_modernbert_forward(), _MODERNBERT_FORWARD_FLOPS, delta=1e7
    )

  def test_counts_attention_of_modernbert_without_torch_formulas(self):
    registry = {
        op: formula
        for op, formula in flop_counter.flop_registry.items()
        if op not in _CPU_SDPA_OPS
    }
    with mock.patch.dict(flop_counter.flop_registry, registry, clear=True):
      self.assertAlmostEqual(
          self._count_modernbert_forward(),
          _MODERNBERT_FORWARD_FLOPS,
          delta=1e7,
      )


if __name__ == "__main__":
  absltest.main()
