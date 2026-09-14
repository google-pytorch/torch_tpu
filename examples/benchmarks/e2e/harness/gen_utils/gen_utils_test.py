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

"""Tests for gen_benchmarks."""

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import torch
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import target as target_lib
from examples.benchmarks.e2e.harness import torch_device_ops
from examples.benchmarks.e2e.harness.gen_utils import gen_utils
from examples.benchmarks.e2e.harness.gen_utils import load_model
from tests import seed_test_utils


class GenUtilsTest(seed_test_utils.RepeatableTest):

  def test_model_entry_benchmark_name(self):
    entry = gen_utils.ModelEntry(
        model_id="foo/bar-baz_123",
        provider="",
        model_type="",
        category="",
        pipeline_tag="",
        downloads=0,
        downloads_all_time=0,
        likes=0,
        created_at="",
        trending_score=0.0,
        params_est=0,
        is_finetune=False,
        base_model="",
        tier="",
    )
    self.assertEqual(
        entry.benchmark_name(is_training=False), "bar_baz_123_forward"
    )
    self.assertEqual(
        entry.benchmark_name(is_training=True), "bar_baz_123_train"
    )
    self.assertEqual(
        entry.benchmark_name(is_training=False, suffix="gen"),
        "bar_baz_123_forward_gen",
    )
    self.assertEqual(
        entry.benchmark_name(is_training=True, suffix="gen"),
        "bar_baz_123_train_gen",
    )

  def test_model_entry_get_extra_config(self):
    entry = gen_utils.ModelEntry(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        provider="",
        model_type="",
        category="",
        pipeline_tag="",
        downloads=0,
        downloads_all_time=0,
        likes=0,
        created_at="",
        trending_score=0.0,
        params_est=0,
        is_finetune=False,
        base_model="",
        tier="",
    )
    config = entry.get_extra_config(is_training=True, suffix="gen")
    self.assertIsNotNone(config)
    self.assertIn("compiled", config.skipped_run_modes)

  def test_iter_models_csv(self):
    models = list(gen_utils.iter_models_csv())
    for model in models:
      self.assertTrue(model.model_id)
      self.assertTrue(model.provider)
      self.assertTrue(model.model_type)
      self.assertTrue(model.category)

  @parameterized.named_parameters(
      ("bf16", target_lib.DType.BF16),
      ("fp32", target_lib.DType.FP32),
  )
  @flagsaver.flagsaver(weights_base_path="")
  def test_default_load_model_dtype(self, dtype: target_lib.DType):
    ctx = context_lib.Context(
        target=target_lib.make_target(target_lib.Platform.CPU, dtype),
        run_scope=context_lib.RunScope.PRESUBMIT,
    )
    # Checked-in vision config whose checkpoint defaults to float32.
    model, inputs = load_model.default_load_model(
        ctx, "microsoft/resnet-50", is_training=False
    )
    want = torch_device_ops.get_torch_dtype(dtype)

    self.assertEqual(next(model.parameters()).dtype, want)
    self.assertEqual(inputs["pixel_values"].dtype, want)

    with torch.inference_mode():
      model(**inputs)


if __name__ == "__main__":
  absltest.main()
