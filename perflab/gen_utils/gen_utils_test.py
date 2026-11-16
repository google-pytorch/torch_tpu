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
from perflab.gen_utils import load_model
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
    self.assertEqual(config.full_scope_batch_size, 1)

  @parameterized.parameters(
      "google/tapas-base-finetuned-sqa",
      "google/tapas-large-finetuned-sqa",
  )
  def test_tapas_sqa_train_benchmarks_use_tapas_load_model(self, model_id: str):
    entry = gen_utils.ModelEntry(
        model_id=model_id,
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
    train_cfg = entry.get_extra_config(is_training=True, suffix="gen")
    forward_cfg = entry.get_extra_config(is_training=False, suffix="gen")
    self.assertIs(train_cfg.model_load_fn, load_model.tapas_load_model)
    self.assertIs(forward_cfg.model_load_fn, load_model.default_load_model)

  def test_full_scope_batch_sizes_match_benchmarks(self):
    benchmark_names = {
        model.benchmark_name(is_training, suffix="gen")
        for model in gen_utils.iter_models_csv()
        for is_training in (False, True)
    }
    for name, batch_size in gen_utils._GEN_FULL_SCOPE_BATCH_SIZES.items():
      with self.subTest(name=name):
        self.assertIn(name, benchmark_names)
        self.assertGreater(batch_size, 1)

  def test_iter_models_csv(self):
    models = list(gen_utils.iter_models_csv())
    for model in models:
      self.assertTrue(model.model_id)
      self.assertTrue(model.provider)
      self.assertTrue(model.model_type)
      self.assertTrue(model.category)


if __name__ == "__main__":
  absltest.main()
