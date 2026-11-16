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

"""Unit tests for distributed Meta Llama 8B model benchmark."""

from unittest import mock

from absl.testing import absltest
from perflab import context as context_lib
from perflab import registry as registry_lib
from perflab import target as target_lib
from perflab.distributed_models import llama
from tests import seed_test_utils


class DistributedLlamaBenchmarkTest(seed_test_utils.RepeatableTest):

  def test_meta_llama_8b_registered(self):
    self.assertIn("meta_llama_8b_forward", registry_lib.REGISTRY)
    spec = registry_lib.REGISTRY["meta_llama_8b_forward"]
    self.assertEqual(spec.name, "meta_llama_8b_forward")

  def test_meta_llama_8b_replica_spans_up_to_its_kv_heads(self):
    # The model is split across gcd(world size, 8 KV heads) devices.
    spec = registry_lib.REGISTRY["meta_llama_8b_forward"]
    for world_size, devices_per_replica in ((4, 4), (8, 8), (16, 8), (12, 4)):
      with self.subTest(world_size=world_size):
        self.assertEqual(
            spec.devices_per_replica(world_size), devices_per_replica
        )

  @mock.patch.object(llama, "_load_meta_llama", return_value=("model", ("x",)))
  def test_meta_llama_8b_supports_gpu_and_tpu_platforms(self, mock_load):
    for platform in (
        target_lib.Platform.V7_2X2X1,
        target_lib.Platform.V7_2X2X2,
        target_lib.Platform.B200_4,
        target_lib.Platform.B200_8,
    ):
      ctx = context_lib.Context(
          target=target_lib.make_target(platform),
          run_scope=context_lib.RunScope.PRESUBMIT,
      )
      model, inputs, kwargs = llama.meta_llama_8b_forward(ctx)
      self.assertEqual(model, "model")
      self.assertEqual(inputs, ("x",))
      self.assertEqual(kwargs, {})
    self.assertEqual(mock_load.call_count, 4)

  def test_meta_llama_8b_rejects_single_device_platforms(self):
    for platform in (target_lib.Platform.V7_1X1X1, target_lib.Platform.B200_1):
      ctx = context_lib.Context(
          target=target_lib.make_target(platform),
          run_scope=context_lib.RunScope.PRESUBMIT,
      )
      with self.assertRaises(  # ASSERT_RAISES_OK=Unsupported platform check.
          target_lib.UnsupportedBenchmark
      ):
        llama.meta_llama_8b_forward(ctx)


if __name__ == "__main__":
  absltest.main()
