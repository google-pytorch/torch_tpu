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

"""Generates benchmarks from models.csv."""

import csv
import dataclasses
import io
import re
from typing import Any, Callable, Iterator

import torch
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness import loss as loss_lib
from examples.benchmarks.e2e.harness.gen_utils import load_model
from examples.benchmarks.e2e.harness.gen_utils import models_csv


@dataclasses.dataclass(frozen=True)
class ModelEntry:
  """Represents a model entry loaded from the models.csv dataset."""

  model_id: str
  provider: str
  model_type: str
  pipeline_tag: str
  downloads: int
  downloads_all_time: int
  likes: int
  created_at: str
  trending_score: float
  params_est: int
  is_finetune: bool
  base_model: str
  tier: str

  def test_case_name(self, is_training, suffix=None):
    """Generates a standardized test case name for this model."""
    name_part = self.model_id.split("/")[-1]
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", name_part.lower()).strip("_")
    mode = "train" if is_training else "inference"
    parts = [clean_name, mode]
    if suffix:
      parts.append(suffix)
    return "_".join(parts)

  def get_extra_config(
      self, is_training: bool, suffix: str | None = None
  ) -> "ExtraConfigs":
    """Retrieves additional configuration for this model's test case, if any."""
    case_name = self.test_case_name(is_training, suffix)
    return _GEN_MODEL_CONFIGS.get(case_name, ExtraConfigs())


def iter_models_csv() -> Iterator[ModelEntry]:
  """Returns an iterator of ModelEntry over the models.csv file."""
  csv_content = models_csv.MODELS_CSV.strip()
  reader = csv.DictReader(io.StringIO(csv_content))
  for row in reader:
    yield ModelEntry(
        model_id=row["model_id"],
        provider=row["provider"],
        model_type=row["model_type"],
        pipeline_tag=row["pipeline_tag"],
        downloads=int(row["downloads"]),
        downloads_all_time=int(row["downloads_all_time"]),
        likes=int(row["likes"]),
        created_at=row["created_at"],
        trending_score=float(row["trending_score"]),
        params_est=int(row["params_est"]),
        is_finetune=row["is_finetune"] == "True",
        base_model=row["base_model"],
        tier=row["tier"],
    )


# There may be extra per-case configuration needed for generated benchmarks that aren't
# captured in any of the static data assets we generate against. These extra configs will
# just be maintained manually. They are opt in, meaning if no config is present the model will
# still be run as is. These are keyed based on the output of the test_case_name() method, which match actual test names in sponge.
@dataclasses.dataclass(frozen=True)
class ExtraConfigs:
  """Extra configuration for a generated benchmark."""

  # What run modes should this benchmark be skipped for.
  skipped_run_modes: set[str] = dataclasses.field(default_factory=set)
  # Additional arguments to pass to the step factory.
  step_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  # The function to use to load the model.
  model_load_fn: Callable[
      [context_lib.Context, str, bool], tuple[torch.nn.Module, dict[str, Any]]
  ] = load_model.default_load_model


# pyformat: disable
_GEN_MODEL_CONFIGS_RAW: dict[str, dict[str, Any]] = {
    "albert_base_v2_finetuned_squad_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "all_minilm_l6_v2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "all_mpnet_base_v2_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "all_mpnet_base_v2_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "altclip_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "altclip_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "aristo_roberta_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "aristo_roberta_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ast_finetuned_audioset_10_10_0_4593_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ast_finetuned_audioset_10_10_0_4593_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "audio_flamingo_3_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "audio_flamingo_3_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "awesome_fb_model_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "awesome_fb_model_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bark_small_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bark_small_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bart_large_mnli_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bart_large_mnli_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "beit_base_finetuned_ade_640_640_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "beit_base_finetuned_ade_640_640_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert2bert_base_arxiv_titlegen_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_base_uncased_finetuned_semeval2020_task4a_append_e2_b32_l5e5_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_base_uncased_finetuned_semeval2020_task4a_append_e2_b32_l5e5_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_base_uncased_finetuned_semeval2020_task4b_base_e2_b32_l3e5_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_base_uncased_finetuned_semeval2020_task4b_base_e2_b32_l3e5_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_base_uncased_finetuned_swag_e1_b16_l5e5_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_base_uncased_finetuned_swag_e1_b16_l5e5_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bert_large_uncased_whole_word_masking_finetuned_squad_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bge_m3_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bge_multilingual_gemma2_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "bge_small_en_v1_5_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "chinese_clip_vit_base_patch16_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clap_htsat_fused_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clap_htsat_fused_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clip_vit_base_patch32_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clip_vit_base_patch32_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clipseg_rd64_refined_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clipseg_rd64_refined_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "coco_panoptic_eomt_large_640_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "coco_panoptic_eomt_large_640_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "colpali_v1_3_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "colqwen2_v1_0_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "contriever_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "convnextv2_tiny_1k_224_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "convnextv2_tiny_1k_224_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "cosmos3_nano_gptq_4bit_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "cosmos3_nano_gptq_4bit_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "csm_1b_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "csm_1b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "cuneiformbase_400m_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "cuneiformbase_400m_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_cifar10_32_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_cifar10_32_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deberta_base_mnli_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deberta_v3_base_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "decision_transformer_gym_hopper_medium_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "decision_transformer_gym_hopper_medium_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deformable_detr_doclaynet_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deformable_detr_doclaynet_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deplot_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deplot_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "depth_anything_v2_small_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "depth_anything_v2_small_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "depthpro_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "depthpro_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "detr_resnet_50_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "detr_resnet_50_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "dinov2_small_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "dinov2_with_registers_base_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "dinov2_with_registers_base_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "dinov3_vitl16_pretrain_lvd1689m_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "distilbert_base_cased_distilled_squad_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "distilbert_base_uncased_finetuned_sst_2_english_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "docling_layout_heron_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "docling_layout_heron_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "dpt_hybrid_midas_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "dpt_hybrid_midas_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "e5_omni_7b_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "e5_omni_7b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "edgetam_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "edgetam_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "fastspeech2_conformer_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "fastspeech2_conformer_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "fastspeech2_conformer_with_hifigan_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "fastspeech2_conformer_with_hifigan_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "finbert_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "flan_t5_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_12b_it_assistant_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_12b_it_assistant_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_12b_it_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_12b_it_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_31b_it_assistant_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_31b_it_assistant_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "git_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "git_large_vqav2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "glm_image_sdnq_4bit_dynamic_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "glm_image_sdnq_4bit_dynamic_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "glm_ocr_train_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "glpn_kitti_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "glpn_kitti_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "glucose_base_ja_v2_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "granite_embedding_small_english_r2_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "granite_embedding_small_english_r2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_speech_3_3_2b_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_speech_3_3_2b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_timeseries_patchtsmixer_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_timeseries_patchtsmixer_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_timeseries_patchtst_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_timeseries_patchtst_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "grounding_dino_base_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
            "compiled",
        },
    },
    "grounding_dino_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "grounding_dino_tiny_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "grounding_dino_tiny_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gte_modernbert_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "gte_modernbert_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "hf_seamless_m4t_medium_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "higgs_tts_2_3b_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "higgs_tts_2_3b_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ijepa_vith14_1k_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "janus_pro_1b_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "janus_pro_1b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "klue_roberta_large_copa_finetuned_v1_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "klue_roberta_large_copa_finetuned_v1_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "kosmos_2_patch14_224_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "kosmos_2_patch14_224_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "lahja_sa_ahmad_v1_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "lahja_sa_ahmad_v1_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "layoutlm_document_qa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "lightglue_superpoint_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "lightglue_superpoint_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "llava_1_5_7b_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "llava_1_5_7b_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "llava_next_video_7b_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "llava_next_video_7b_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "llmdet_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "llmdet_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "manga_ocr_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mask2former_swin_large_ade_semantic_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mask2former_swin_large_ade_semantic_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mbart_ru_sum_gazeta_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "metaclip_2_worldwide_giant_378_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "metaclip_2_worldwide_giant_378_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mgp_str_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mgp_str_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "minit2i_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "minit2i_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mmarco_mminilmv2_l12_h384_v1_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mmarco_mminilmv2_l12_h384_v1_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "modernbert_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "modernbert_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "ms_marco_minilm_l6_v2_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ms_marco_minilm_l6_v2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mt5_multilingual_xlsum_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mt5_zh_ja_en_trimmed_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "multilingual_e5_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "music_flamingo_2601_hf_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "music_flamingo_2601_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_medium_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_medium_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_melody_bella_ciao_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_melody_bella_ciao_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_melody_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_melody_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_small_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "musicgen_small_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nases_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nemotron_3_5_asr_streaming_0_6b_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nemotron_3_5_asr_streaming_0_6b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nemotron_3_embed_1b_bf16_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nemotron_3_embed_1b_bf16_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nllb_200_distilled_600m_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_gen_anime_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_gen_anime_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_image_detection_large_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_image_detection_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_image_detection_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_image_detector_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_image_detector_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "omdet_turbo_swin_tiny_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "omdet_turbo_swin_tiny_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "oneformer_ade20k_swin_large_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "oneformer_ade20k_swin_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "opus_mt_nl_en_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "owlv2_base_patch16_ensemble_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "owlv2_base_patch16_ensemble_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "owlvit_base_patch32_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "owlvit_base_patch32_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "parakeet_ctc_1_1b_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "parakeet_ctc_1_1b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "pegasus_xsum_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "pix2struct_tiny_random_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "pix2struct_tiny_random_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "pp_doclayoutv3_safetensors_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "pp_doclayoutv3_safetensors_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "privacy_filter_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "privacy_filter_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "prompt_depth_anything_vits_hf_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "prompt_depth_anything_vits_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "prompt_guard_86m_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qanlu_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_5_omni_3b_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_5_omni_3b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_5_vl_7b_instruct_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_5_vl_7b_instruct_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_audio_7b_instruct_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_audio_7b_instruct_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_vl_2b_instruct_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_5_9b_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_5_9b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_asr_1_7b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_vl_8b_instruct_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_vl_8b_instruct_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "resnet_50_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "resnet_50_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "roberta_base_go_emotions_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "roberta_large_finetuned_race_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "roberta_large_finetuned_race_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "rtdetr_r101vd_coco_o365_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "rtdetr_r101vd_coco_o365_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "rut5_base_sum_gazeta_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam2_hiera_base_plus_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam2_hiera_base_plus_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam3_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam3_litetext_s0_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam3_litetext_s0_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam3_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam_hq_vit_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam_hq_vit_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam_vit_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam_vit_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sapiens2_pose_0_4b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sdxl_detector_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sdxl_detector_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "segformer_b0_finetuned_ade_512_512_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "segformer_b0_finetuned_ade_512_512_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "siglip2_base_patch16_naflex_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "siglip2_base_patch16_naflex_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "siglip_so400m_patch14_384_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "siglip_so400m_patch14_384_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "splinter_base_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "splinter_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "superpoint_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "superpoint_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swin2sr_classical_sr_x2_64_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swin2sr_classical_sr_x2_64_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swinv2_tiny_patch4_window16_256_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swinv2_tiny_patch4_window16_256_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "t5_small_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "table_transformer_structure_recognition_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "table_transformer_structure_recognition_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "tapas_large_finetuned_sqa_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "tapas_large_finetuned_sqa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "text2vec_base_chinese_paraphrase_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "textnet_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "textnet_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "time_anchor_modernbert_32m_inference_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "time_anchor_modernbert_32m_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "time_series_transformer_tourism_monthly_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "time_series_transformer_tourism_monthly_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "timesfm_2_5_200m_transformers_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "timesfm_2_5_200m_transformers_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "timesformer_base_finetuned_k600_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "timesformer_base_finetuned_k600_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "twitter_xlm_roberta_base_sentiment_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vibevoice_asr_hf_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "vibevoice_asr_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "videomae_base_finetuned_kinetics_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "videomae_base_finetuned_kinetics_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "videomae_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "videomae_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vietai_nlp_itn_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vilt_b32_finetuned_vqa_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vilt_b32_finetuned_vqa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vit_base_patch16_224_in21k_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vitpose_plus_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vitpose_plus_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vivit_b_16x2_kinetics400_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vivit_b_16x2_kinetics400_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vjepa2_vitl_fpc64_256_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vjepa2_vitl_fpc64_256_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "w2v_bert_2_0_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "w2v_bert_2_0_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_base_960h_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_base_960h_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_base_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_large_robust_12_ft_emotion_msp_dim_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_large_robust_12_ft_emotion_msp_dim_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_lg_xlsr_en_speech_emotion_recognition_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_lg_xlsr_en_speech_emotion_recognition_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_xlsr_greek_speech_emotion_recognition_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_xlsr_greek_speech_emotion_recognition_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_xlsr_japanese_speech_emotion_recognition_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_xlsr_japanese_speech_emotion_recognition_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wavlm_base_plus_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wavlm_base_plus_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "whisper_large_v3_turbo_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "whisper_large_v3_turbo_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xclip_base_patch32_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xclip_base_patch32_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xcodec2_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xcodec2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xlm_roberta_large_it_mnli_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xlm_roberta_large_ner_hrl_inference_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "xlm_roberta_large_qa_multilingual_finedtuned_ru_train_gen": {
        "step_kwargs": {
            "compute_loss": loss_lib.dummy_compute_loss,
        },
    },
    "xlm_roberta_large_xnli_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "yolos_small_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "yolos_small_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "zeroaraelectra_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "zoedepth_nyu_kitti_inference_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "zoedepth_nyu_kitti_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
}
# pyformat: enable


_GEN_MODEL_CONFIGS: dict[str, ExtraConfigs] = {
    k: ExtraConfigs(**v) for k, v in _GEN_MODEL_CONFIGS_RAW.items()
}
