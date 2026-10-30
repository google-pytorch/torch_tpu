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

"""Checks that the generated benchmarks stay consistent and smoke-testable.

The FakeTensorMode CPU smoke test (`perflab:torch_benchmark_cpu_smoke_test`)
builds every generated benchmark whose model config is in-tree. These checks
keep that coverage from shrinking silently and catch configuration mistakes
that otherwise only show up in the nightly benchmark runs.
"""

from collections.abc import Set
from absl.testing import absltest
from perflab import common
from perflab import mode as mode_lib
from perflab import target as target_lib
from perflab.gen_utils import gen_utils
from perflab.models import gen_benchmarks
from tests import seed_test_utils

# Models of generated benchmarks that run on an accelerator but whose configs
# are only on GCS, so the smoke test, which has no network access, cannot build
# them. Add the `config.json` of a new model to
# `google/examples/huggingface_transformers/model_configs` rather than adding
# the model here.
_GCS_ONLY_MODEL_IDS = frozenset({
    "Akari/albert-base-v2-finetuned-squad",
    "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru",
    "Alibaba-NLP/gte-modernbert-base",
    "AmazonScience/qanlu",
    "BAAI/AltCLIP",
    "BAAI/bge-multilingual-gemma2",
    "Callidior/bert2bert-base-arxiv-titlegen",
    "Davlan/distilbert-base-multilingual-cased-ner-hrl",
    "Davlan/xlm-roberta-large-ner-hrl",
    "ELiRF/NASES",
    "IlyaGusev/mbart_ru_sum_gazeta",
    "IlyaGusev/rut5_base_sum_gazeta",
    "Jean-Baptiste/camembert-ner",
    "Jean-Baptiste/roberta-large-ner-english",
    "Jiva/xlm-roberta-large-it-mnli",
    "K024/mt5-zh-ja-en-trimmed",
    "KheireddineDaouadi/ZeroAraElectra",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "SamLowe/roberta-base-go_emotions",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "czczup/textnet-base",
    "datnth1709/VietAI-NLP-ITN",
    "deepseek-community/Janus-Pro-1B",
    "dslim/bert-base-NER",
    "facebook/esm2_t33_650M_UR50D",
    "facebook/sapiens2-pose-0.4b",
    "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
    "google/flan-t5-base",
    "microsoft/deberta-v3-base",
    "microsoft/git-base",
    "microsoft/git-large-vqav2",
    "nineninesix/gepard-1.0",
    "pkshatech/GLuCoSE-base-ja-v2",
    "shibing624/text2vec-base-chinese-paraphrase",
    "tau/splinter-base",
})

# The run modes of the benchmark runners on TPUs and GPUs.
_ACCELERATOR_RUN_MODES = frozenset(
    run_mode.value
    for device_kind in (target_lib.DeviceKind.TPU, target_lib.DeviceKind.CUDA)
    for run_mode in mode_lib.modes_for(mode_lib.Framework.TORCH, device_kind)
)


def _generated_benchmark_names() -> Set[str]:
  return {
      model.benchmark_name(is_training, suffix="gen")
      for model in gen_utils.iter_models_csv()
      for is_training in (False, True)
  }


def _runs_on_an_accelerator(model: gen_utils.ModelEntry) -> bool:
  """Returns whether a generated benchmark of `model` runs on TPUs or GPUs."""
  return any(
      _ACCELERATOR_RUN_MODES
      - model.get_extra_config(is_training, suffix="gen").skipped_run_modes
      for is_training in (False, True)
  )


class BenchmarkContractTest(seed_test_utils.RepeatableTest):

  def test_generated_benchmarks_that_run_have_in_tree_configs(self):
    gcs_only = {
        model.model_id
        for model in gen_utils.iter_models_csv()
        if _runs_on_an_accelerator(model)
        and not gen_benchmarks.has_in_tree_config(model.model_id)
    }
    self.assertEmpty(
        sorted(gcs_only - _GCS_ONLY_MODEL_IDS),
        "the CPU smoke test cannot build these models: add their config.json"
        " to google/examples/huggingface_transformers/model_configs",
    )
    self.assertEmpty(
        sorted(_GCS_ONLY_MODEL_IDS - gcs_only),
        "these models have in-tree configs or no longer run: remove them from"
        " _GCS_ONLY_MODEL_IDS",
    )

  def test_extra_configs_name_generated_benchmarks(self):
    # A misspelled or stale name silently stops applying its skips, step
    # arguments and batch size to the benchmark.
    self.assertContainsSubset(
        gen_utils._GEN_MODEL_CONFIGS,  # pylint: disable=protected-access
        _generated_benchmark_names(),
    )

  def test_skipped_run_modes_are_run_modes(self):
    run_modes = {run_mode.value for run_mode in common.RunMode}
    for name, config in gen_utils._GEN_MODEL_CONFIGS.items():  # pylint: disable=protected-access
      with self.subTest(name):
        self.assertContainsSubset(config.skipped_run_modes, run_modes)


if __name__ == "__main__":
  absltest.main()
