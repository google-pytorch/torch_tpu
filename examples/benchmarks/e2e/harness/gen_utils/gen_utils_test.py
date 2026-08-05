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
from examples.benchmarks.e2e.harness.gen_utils import gen_utils
from tests import seed_test_utils


class GenUtilsTest(seed_test_utils.RepeatableTest):

  def test_model_entry_benchmark_name(self):
    entry = gen_utils.ModelEntry(
        model_id="foo/bar-baz_123",
        provider="",
        model_type="",
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
        model_id="albert-base-v2-finetuned-squad",
        provider="",
        model_type="",
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


if __name__ == "__main__":
  absltest.main()
