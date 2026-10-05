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

import types

from absl.testing import absltest
from absl.testing import parameterized
import torch
from perflab.gen_utils import gen_utils
from tests import seed_test_utils


class GenUtilsTest(seed_test_utils.RepeatableTest):

  @parameterized.named_parameters(
      (
          "last_hidden_state",
          lambda x: types.SimpleNamespace(last_hidden_state=x),
      ),
      ("pooler_output", lambda x: types.SimpleNamespace(pooler_output=x)),
      ("tuple", lambda x: (x,)),
      ("tensor", lambda x: x),
      ("dict", lambda x: {"start_logits": x}),
  )
  def test_dummy_compute_loss_returns_mean(self, model):
    x = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    loss = gen_utils.dummy_compute_loss(model, (x,), {})
    self.assertEqual(loss.item(), 4.0)

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


if __name__ == "__main__":
  absltest.main()
