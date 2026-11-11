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

"""Unit tests for the distributed Qwen3.5-397B-A17B benchmark."""

from absl.testing import absltest
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.distributed_models import qwen
from tests import seed_test_utils


class DistributedQwenBenchmarkTest(seed_test_utils.RepeatableTest):

  def test_qwen3_5_397b_registered(self):
    self.assertIn("qwen3_5_397b_a17b_forward", registry_lib.REGISTRY)

  def test_qwen3_5_397b_replica_spans_all_devices(self):
    # The model is split across all devices, so there is one replica.
    spec = registry_lib.REGISTRY["qwen3_5_397b_a17b_forward"]
    for world_size in (8, 16, 32):
      with self.subTest(world_size=world_size):
        self.assertEqual(spec.devices_per_replica(world_size), world_size)

  def test_qwen3_5_397b_rejects_unsupported_platforms(self):
    for platform in (target_lib.Platform.V7_2X2X1, target_lib.Platform.B200_4):
      ctx = context_lib.Context(
          target=target_lib.make_target(platform),
          run_scope=context_lib.RunScope.PRESUBMIT,
      )
      with self.subTest(platform=platform):
        with self.assertRaises(  # ASSERT_RAISES_OK=Unsupported platform check.
            target_lib.UnsupportedBenchmark
        ):
          qwen.qwen3_5_397b_a17b_forward(ctx)

  def test_qwen3_5_397b_requires_initialized_process_group(self):
    ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.B200_8),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    with self.assertRaisesRegex(  # ASSERT_RAISES_OK=Process group check.
        target_lib.UnsupportedBenchmark, "initialized torch.distributed"
    ):
      qwen.qwen3_5_397b_a17b_forward(ctx)


if __name__ == "__main__":
  absltest.main()
