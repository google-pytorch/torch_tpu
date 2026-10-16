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
import importlib
import io
import os
import pathlib
import re
from typing import Any, Callable, Iterator

import torch
from perflab import context as context_lib
from perflab.gen_utils import load_model

# Called as `model_load_fn(ctx, model_id, is_training, full_scope_batch_size)`.
ModelLoadFn = Callable[
    [context_lib.Context, str, bool, int],
    tuple[torch.nn.Module, dict[str, Any]],
]


def _get_models_csv_file():
  """Internalizes different ways to locate models.csv."""
  # 1. Try importlib (Standard for installed package/google3)
  try:
    csv_file = importlib.resources.files("torch_tpu").joinpath(
        "perflab/gen_utils/models.csv"
    )
    if csv_file.is_file():
      return csv_file
  except Exception:
    pass

  # 2. Try Bazel runfiles (For tests/dev in Bazel)
  runfiles_dir = os.environ.get("RUNFILES_DIR")  # PY_ENVIRON_OK=benchmark
  if not runfiles_dir:
    runfiles_dir = os.environ.get("TEST_SRCDIR")  # PY_ENVIRON_OK=benchmark
  if runfiles_dir:
    path_main = (
        pathlib.Path(runfiles_dir) / "_main" / "perflab/gen_utils/models.csv"
    )
    if path_main.is_file():
      return path_main

    path_torch_tpu = (
        pathlib.Path(runfiles_dir)
        / "torch_tpu"
        / "perflab/gen_utils/models.csv"
    )
    if path_torch_tpu.is_file():
      return path_torch_tpu

  # 3. Try local path (Fallback)
  local_path = pathlib.Path("perflab/gen_utils/models.csv")
  if local_path.is_file():
    return local_path

  raise FileNotFoundError("models.csv not found.")


def dummy_compute_loss(model, input_args, input_kwargs) -> torch.Tensor:
  """A dummy compute loss function for models that do not return a loss naturally."""
  out = model(*input_args, **input_kwargs)
  if hasattr(out, "last_hidden_state"):
    return out.last_hidden_state.mean()
  if hasattr(out, "pooler_output"):
    return out.pooler_output.mean()
  if (
      isinstance(out, (tuple, list))
      and len(out) > 0
      and torch.is_tensor(out[0])
  ):
    return out[0].mean()
  if torch.is_tensor(out):
    return out.mean()
  if isinstance(out, dict):
    for v in out.values():
      if torch.is_tensor(v):
        return v.mean()
  raise TypeError(
      "dummy_compute_loss could not find a tensor to average in"
      f" {type(out).__name__}"
  )


@dataclasses.dataclass(frozen=True)
class ModelEntry:
  """Represents a model entry loaded from the models.csv dataset."""

  model_id: str
  provider: str
  model_type: str
  category: str
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

  def benchmark_name(self, is_training, suffix=None):
    """Generates a standardized test case name for this model."""
    name_part = self.model_id.split("/")[-1]
    clean_name = re.sub(r"[^a-zA-Z0-9]+", "_", name_part.lower()).strip("_")
    mode = "train" if is_training else "forward"
    parts = [clean_name, mode]
    if suffix:
      parts.append(suffix)
    return "_".join(parts)

  def get_extra_config(
      self, is_training: bool, suffix: str | None = None
  ) -> "ExtraConfigs":
    """Retrieves additional configuration for this model's test case, if any."""
    case_name = self.benchmark_name(is_training, suffix)
    return _GEN_MODEL_CONFIGS.get(case_name, ExtraConfigs())


def iter_models_csv() -> Iterator[ModelEntry]:
  """Returns an iterator of ModelEntry over the models.csv file."""
  csv_file = _get_models_csv_file()

  csv_content = csv_file.read_text().strip()

  reader = csv.DictReader(io.StringIO(csv_content))
  for row in reader:
    yield ModelEntry(
        model_id=row["model_id"],
        provider=row["provider"],
        model_type=row["model_type"],
        category=row["category"],
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
# just be maintained manually. They are opt in, meaning if no config is present the model
# will still be run as is. These are keyed based on the output of the benchmark_name()
# method, which match actual test names in sponge.
@dataclasses.dataclass(frozen=True)
class ExtraConfigs:
  """Extra configuration for a generated benchmark."""

  # What run modes should this benchmark be skipped for.
  skipped_run_modes: set[str] = dataclasses.field(default_factory=set)
  # Additional arguments to pass to the step factory.
  step_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  # The function to use to load the model.
  model_load_fn: ModelLoadFn = load_model.default_load_model
  # Batch size for the full run scope (see `_GEN_FULL_SCOPE_BATCH_SIZES`).
  full_scope_batch_size: int = 1


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
    "all_mpnet_base_v2_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "all_mpnet_base_v2_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "altclip_m9_train_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "aristo_roberta_forward_gen": {
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
    "ast_finetuned_audioset_10_10_0_4593_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "audio_flamingo_3_hf_forward_gen": {
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
    "bark_small_forward_gen": {
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
    "beit_base_finetuned_ade_640_640_forward_gen": {
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
    "bert_base_uncased_finetuned_semeval2020_task4a_append_e2_b32_l5e5_forward_gen": {
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
    "bert_base_uncased_finetuned_semeval2020_task4b_base_e2_b32_l3e5_forward_gen": {
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
    "bert_base_uncased_finetuned_swag_e1_b16_l5e5_forward_gen": {
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
    "bert_large_uncased_whole_word_masking_squad2_train_gen": {
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
            "compute_loss": dummy_compute_loss,
        },
    },
    "bge_reranker_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bge_small_en_v1_5_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "bigbird_pegasus_large_arxiv_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "bigbird_pegasus_large_arxiv_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "blip_vqa_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "blip_vqa_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "blip_vqa_capfilt_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "chinese_clip_vit_base_patch16_train_gen": {
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
    "clap_htsat_unfused_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "clip4clip_webvid150k_train_gen": {
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
    "clipseg_rd64_refined_forward_gen": {
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
    "coco_panoptic_eomt_large_640_forward_gen": {
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
    "colmodernvbert_merged_st_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "colpali_v1_3_hf_forward_gen": {
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
            "eager_default",
            "eager_optimized",
        },
    },
    "convnextv2_base_22k_384_train_gen": {
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
    "cosmos3_nano_gptq_4bit_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "cross_encoder_mmarco_german_distilbert_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "ddpm_celebahq_256_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_celebahq_256_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_cifar10_32_dot_in_name_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_cifar10_32_dot_in_name_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_cifar10_32_forward_gen": {
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
    "ddpm_ema_celebahq_256_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ddpm_ema_celebahq_256_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deberta_v3_base_train_gen": {
        "step_kwargs": {
            "compute_loss": dummy_compute_loss,
        },
    },
    "decision_transformer_gym_hopper_medium_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
        },
    },
    "decision_transformer_gym_hopper_medium_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "deformable_detr_doclaynet_forward_gen": {
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
    "deplot_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
        },
    },
    "deplot_train_gen": {
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
    "depthpro_hf_forward_gen": {
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
    "detr_resnet_50_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "dinov2_small_train_gen": {
        "step_kwargs": {'compute_loss': dummy_compute_loss},
    },
    "dinov2_with_registers_base_train_gen": {
        "step_kwargs": {'compute_loss': dummy_compute_loss},
    },
    "dinov3_vitl16_pretrain_lvd1689m_train_gen": {
        "step_kwargs": {'compute_loss': dummy_compute_loss},
    },
    "distilbert_base_cased_distilled_squad_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "distiluse_base_multilingual_cased_v2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "docling_layout_heron_forward_gen": {
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
    "donut_base_finetuned_docvqa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_optimized",
        },
    },
    "donut_base_finetuned_docvqa_train_gen": {
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
    "dse_qwen2_2b_mrl_v1_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "dse_qwen2_2b_mrl_v1_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "edgetam_hf_forward_gen": {
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
    "electra_large_discriminator_squad2_512_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "fastspeech2_conformer_forward_gen": {
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
    "fastspeech2_conformer_with_hifigan_forward_gen": {
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
    "gemma_4_12b_it_assistant_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_12b_it_forward_gen": {
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
    "gemma_4_31b_it_assistant_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "gemma_4_e4b_it_train_gen": {
        "skipped_run_modes": {
            # TODO(b/568804716): On TPU, the backward of the image-token
            # masked_scatter calls masked_select, whose data-dependent output
            # shape forces a device-to-host copy during the FX trace.
            "compiled",
        },
    },
    "glm_image_sdnq_4bit_dynamic_forward_gen": {
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
            "compiled",
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
            "compute_loss": dummy_compute_loss,
        },
    },
    "granite_embedding_small_english_r2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_speech_3_3_2b_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "granite_timeseries_patchtsmixer_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "granite_timeseries_patchtst_forward_gen": {
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
    "grounding_dino_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "grounding_dino_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "grounding_dino_tiny_forward_gen": {
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
    "gte_modernbert_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "higgs_tts_2_3b_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "higgs_tts_2_3b_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "hubert_large_speech_emotion_recognition_russian_dusha_finetuned_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "if_i_xl_v1_0_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "if_i_xl_v1_0_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "if_ii_l_v1_0_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "if_ii_l_v1_0_train_gen": {
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
    "instructor_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "janus_pro_1b_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "klue_roberta_large_copa_finetuned_v1_forward_gen": {
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
    "kosmos_2_patch14_224_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "lahja_sa_ahmad_v1_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "layoutlm_document_qa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "layoutlm_document_qa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "layoutlm_invoices_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "layoutlmv2_base_uncased_finetuned_docvqa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "layoutlmv2_base_uncased_finetuned_docvqa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ldm_celebahq_256_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "ldm_celebahq_256_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "lightglue_superpoint_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "llava_1_5_7b_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "llava_next_video_7b_hf_forward_gen": {
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
    "llmdet_base_forward_gen": {
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
    "manga_ocr_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "mask2former_swin_large_ade_semantic_forward_gen": {
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
    "mdeberta_v3_base_train_gen": {
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
    "mgp_str_base_forward_gen": {
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
    "minit2i_forward_gen": {
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
    "mit_b2_train_gen": {
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
    "ms_marco_minilm_l6_v2_train_gen": {
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
    "musicgen_medium_forward_gen": {
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
    "musicgen_melody_bella_ciao_forward_gen": {
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
    "musicgen_melody_forward_gen": {
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
    "musicgen_small_forward_gen": {
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
    "nemotron_3_5_asr_streaming_0_6b_forward_gen": {
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
    "nemotron_3_embed_1b_bf16_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "nsfw_gen_anime_forward_gen": {
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
    "nsfw_image_detector_forward_gen": {
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
    "omdet_turbo_swin_tiny_hf_forward_gen": {
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
    "oneformer_ade20k_swin_large_forward_gen": {
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
    "oneformer_ade20k_swin_tiny_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "oneformer_ade20k_swin_tiny_train_gen": {
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
    "owlvit_base_patch32_train_gen": {
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
    "pix2struct_tiny_random_forward_gen": {
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
    "pp_doclayoutv3_safetensors_forward_gen": {
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
    "privacy_filter_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "privacy_filter_train_gen": {
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
    "qanlu_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qnli_electra_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen2_5_omni_3b_forward_gen": {
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
    "qwen2_5_vl_7b_instruct_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "qwen2_5_vl_7b_instruct_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "qwen2_vl_2b_instruct_forward_gen": {
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
    "qwen3_5_9b_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "qwen3_asr_0_6b_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_asr_0_6b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "qwen3_asr_1_7b_forward_gen": {
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
    "qwen3_vl_8b_instruct_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "roberta_base_squad2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "roberta_large_finetuned_csqa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "roberta_large_finetuned_csqa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "roberta_large_finetuned_race_forward_gen": {
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
    "rtdetr_r101vd_coco_o365_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam2_hiera_base_plus_forward_gen": {
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
    "sam3_forward_gen": {
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
    "sam_hq_vit_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "sam_hq_vit_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam_vit_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "sdxl_detector_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "sdxl_detector_train_gen": {
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
    "siglip2_base_patch16_naflex_forward_gen": {
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
    "siglip_so400m_patch14_384_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "speecht5_vc_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "speecht5_vc_train_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "stable_diffusion_safety_checker_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "stable_diffusion_safety_checker_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "stanford_deidentifier_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "stsb_roberta_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "superpoint_forward_gen": {
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
    "swin2sr_classical_sr_x2_64_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "swin2sr_classical_sr_x2_64_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swin2sr_classical_sr_x4_64_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "swin2sr_classical_sr_x4_64_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swin2sr_compressed_sr_x4_48_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "swin2sr_compressed_sr_x4_48_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swin2sr_lightweight_x2_64_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "swin2sr_lightweight_x2_64_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swin2sr_realworld_sr_x4_64_bsrgan_psnr_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "swin2sr_realworld_sr_x4_64_bsrgan_psnr_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "swinv2_tiny_patch4_window16_256_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
            "eager_default",
        },
    },
    "table_transformer_structure_recognition_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "tapas_base_finetuned_wtq_train_gen": {
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
    "textnet_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "time_anchor_modernbert_32m_train_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "time_series_transformer_tourism_monthly_forward_gen": {
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
    "timesfm_2_5_200m_transformers_train_gen": {
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
    "tiny_doc_qa_vision_encoder_decoder_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vibevoice_asr_hf_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "videomae_base_finetuned_kinetics_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "videomae_base_finetuned_kinetics_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "videomae_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "videomae_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "vietai_nlp_itn_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "vilt_b32_finetuned_vqa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "vitpose_plus_base_train_gen": {
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
    "vjepa2_vitl_fpc64_256_train_gen": {
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
    "wav2vec2_base_960h_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_base_forward_gen": {
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
    "wav2vec2_large_robust_12_ft_emotion_msp_dim_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wav2vec2_large_xlsr_53_portuguese_train_gen": {
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
    "wav2vec2_xlsr_greek_speech_emotion_recognition_train_gen": {
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
    "wavlm_base_plus_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "wavlm_base_plus_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "wavlm_emotion_russian_resd_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "wavlm_emotion_russian_resd_train_gen": {
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
    "wmt19_ru_en_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "xclip_base_patch32_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xcodec2_forward_gen": {
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
    "xlm_roberta_large_qa_multilingual_finedtuned_ru_train_gen": {
        "step_kwargs": {
            "compute_loss": dummy_compute_loss,
        },
    },
    "xlm_roberta_large_squad2_train_gen": {
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
    "zoedepth_nyu_kitti_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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


# Full run scope batch sizes for generated benchmarks, keyed like
# `_GEN_MODEL_CONFIGS_RAW`. At batch size 1, these benchmarks spend most of each
# step on host overhead rather than on the device, which hides device and
# compiler regressions. Each value is the smallest power of two (at most 64) for
# which the compiled run mode on v7_1x1x1 spends at least 85% of the step time
# on the device (XProf device time / post-warmup step time), or at least 80% if
# no measured batch size reached 85%. Benchmarks that reach neither (within 64,
# or before larger batch sizes fail) use the measured batch size with the
# highest utilization. Benchmarks not listed use batch size 1. The presubmit
# run scope always uses batch size 1.
# pyformat: disable
_GEN_FULL_SCOPE_BATCH_SIZES: dict[str, int] = {
    # go/keep-sorted start
    "albert_base_v2_finetuned_squad_forward_gen": 16,
    "all_distilroberta_v1_forward_gen": 8,
    "all_distilroberta_v1_train_gen": 4,
    "all_minilm_l6_v2_forward_gen": 32,
    "altclip_forward_gen": 8,
    "altclip_m9_forward_gen": 8,
    "ast_finetuned_audioset_10_10_0_4593_forward_gen": 4,
    "bart_large_cnn_forward_gen": 2,
    "bart_large_cnn_train_gen": 2,
    "bart_large_mnli_forward_gen": 4,
    "bert2bert_base_arxiv_titlegen_forward_gen": 8,
    "bert_base_ner_forward_gen": 16,
    "bert_base_ner_train_gen": 8,
    "bert_base_uncased_forward_gen": 16,
    "bert_base_uncased_train_gen": 4,
    "bert_large_uncased_whole_word_masking_finetuned_squad_forward_gen": 8,
    "bge_reranker_base_forward_gen": 16,
    "bge_small_en_v1_5_forward_gen": 16,
    "blip_image_captioning_base_forward_gen": 8,
    "blip_image_captioning_base_train_gen": 4,
    "blip_vqa_capfilt_large_forward_gen": 16,
    "camembert_ner_forward_gen": 8,
    "camembert_ner_train_gen": 8,
    "chinese_clip_vit_base_patch16_forward_gen": 16,
    "clap_htsat_fused_forward_gen": 16,
    "clip4clip_webvid150k_forward_gen": 64,
    "clip_vit_base_patch32_forward_gen": 64,
    "colqwen2_v1_0_hf_forward_gen": 4,
    "contriever_forward_gen": 16,
    "cross_encoder_mmarco_german_distilbert_base_forward_gen": 16,
    "csm_1b_forward_gen": 64,
    "detr_resnet_50_forward_gen": 64,
    "dinov2_small_forward_gen": 8,
    "dinov2_small_train_gen": 4,
    "dinov2_with_registers_base_forward_gen": 4,
    "dinov2_with_registers_base_train_gen": 2,
    "dinov3_vitl16_pretrain_lvd1689m_forward_gen": 16,
    "dinov3_vitl16_pretrain_lvd1689m_train_gen": 16,
    "distilbert_base_cased_distilled_squad_forward_gen": 32,
    "distilbert_base_multilingual_cased_ner_hrl_forward_gen": 32,
    "distilbert_base_multilingual_cased_ner_hrl_train_gen": 8,
    "distilbert_base_uncased_finetuned_sst_2_english_forward_gen": 16,
    "distilbert_base_uncased_finetuned_sst_2_english_train_gen": 8,
    "distilbert_base_uncased_forward_gen": 16,
    "distilbert_base_uncased_mnli_forward_gen": 16,
    "distilbert_base_uncased_mnli_train_gen": 8,
    "distilbert_base_uncased_train_gen": 8,
    "distiluse_base_multilingual_cased_v2_forward_gen": 32,
    "electra_base_discriminator_forward_gen": 16,
    "electra_base_discriminator_train_gen": 8,
    "electra_large_discriminator_squad2_512_forward_gen": 8,
    "finbert_forward_gen": 16,
    "finbert_train_gen": 4,
    "flan_t5_base_forward_gen": 8,
    "french_camembert_postag_model_forward_gen": 16,
    "french_camembert_postag_model_train_gen": 8,
    "git_base_forward_gen": 8,
    "git_base_vqav2_forward_gen": 4,
    "git_large_vqav2_forward_gen": 2,
    "glm_ocr_forward_gen": 16,
    "glpn_kitti_forward_gen": 32,
    "glucose_base_ja_v2_forward_gen": 8,
    "glucose_base_ja_v2_train_gen": 2,
    "gpt2_forward_gen": 4,
    "granite_timeseries_patchtsmixer_forward_gen": 64,
    "hubert_large_speech_emotion_recognition_russian_dusha_finetuned_forward_gen": 64,
    "ijepa_vith14_1k_forward_gen": 8,
    "inclusively_reformulation_it5_forward_gen": 8,
    "indonesian_roberta_base_posp_tagger_forward_gen": 16,
    "indonesian_roberta_base_posp_tagger_train_gen": 8,
    "instructor_large_forward_gen": 8,
    "lahja_sa_ahmad_v1_forward_gen": 64,
    "layoutlm_invoices_forward_gen": 16,
    "manga_ocr_base_forward_gen": 64,
    "mbart_large_50_many_to_many_mmt_forward_gen": 2,
    "mbart_large_50_many_to_many_mmt_train_gen": 2,
    "mbart_ru_sum_gazeta_forward_gen": 2,
    "metaclip_2_worldwide_giant_378_forward_gen": 2,
    "mit_b2_forward_gen": 32,
    "mmarco_mminilmv2_l12_h384_v1_forward_gen": 16,
    "ms_marco_minilm_l6_v2_forward_gen": 32,
    "mt5_multilingual_xlsum_forward_gen": 8,
    "mt5_multilingual_xlsum_train_gen": 2,
    "mt5_zh_ja_en_trimmed_forward_gen": 8,
    "multilingual_e5_large_forward_gen": 8,
    "nases_forward_gen": 8,
    "nli_minilm2_l6_h768_forward_gen": 16,
    "nli_minilm2_l6_h768_train_gen": 8,
    "nllb_200_distilled_600m_forward_gen": 2,
    "nllb_200_distilled_600m_train_gen": 2,
    "nsfw_image_detection_forward_gen": 32,
    "nsfw_image_detection_large_forward_gen": 32,
    "opt_125m_forward_gen": 2,
    "opt_125m_train_gen": 2,
    "opus_mt_nl_en_forward_gen": 16,
    "opus_mt_nl_en_train_gen": 16,
    "parakeet_ctc_1_1b_forward_gen": 64,
    "pegasus_xsum_forward_gen": 8,
    "pegasus_xsum_train_gen": 4,
    "prompt_guard_86m_train_gen": 4,
    "punctuate_all_forward_gen": 8,
    "punctuate_all_train_gen": 8,
    "qanlu_forward_gen": 16,
    "qnli_electra_base_forward_gen": 16,
    "resnet_50_forward_gen": 64,
    "roberta_base_go_emotions_forward_gen": 8,
    "roberta_base_squad2_forward_gen": 16,
    "roberta_large_forward_gen": 4,
    "roberta_large_ner_english_forward_gen": 4,
    "roberta_large_ner_english_train_gen": 2,
    "roberta_large_train_gen": 2,
    "rut5_base_sum_gazeta_forward_gen": 8,
    "segformer_b0_finetuned_ade_512_512_forward_gen": 64,
    "splinter_base_forward_gen": 16,
    "stanford_deidentifier_base_forward_gen": 16,
    "stsb_roberta_large_forward_gen": 8,
    "t5_small_forward_gen": 16,
    "t5_small_train_gen": 4,
    "table_transformer_structure_recognition_forward_gen": 64,
    "tapex_base_finetuned_wikisql_forward_gen": 4,
    "tapex_base_finetuned_wikisql_train_gen": 4,
    "tapex_large_finetuned_wtq_forward_gen": 2,
    "tapex_large_finetuned_wtq_train_gen": 2,
    "text2vec_base_chinese_paraphrase_forward_gen": 4,
    "text_summarization_forward_gen": 32,
    "text_summarization_train_gen": 4,
    "timesformer_base_finetuned_k600_forward_gen": 2,
    "tiny_qwen2forcausallm_2_5_forward_gen": 4,
    "twitter_roberta_base_sentiment_latest_forward_gen": 16,
    "twitter_roberta_base_sentiment_latest_train_gen": 4,
    "twitter_xlm_roberta_base_sentiment_forward_gen": 8,
    "vietai_nlp_itn_forward_gen": 16,
    "vit_base_patch16_224_in21k_forward_gen": 32,
    "vitpose_plus_base_forward_gen": 64,
    "w2v_bert_2_0_forward_gen": 32,
    "wav2vec2_large_robust_12_ft_emotion_msp_dim_forward_gen": 64,
    "wav2vec2_large_xlsr_53_portuguese_forward_gen": 64,
    "wmt19_ru_en_forward_gen": 4,
    "xclip_base_patch32_forward_gen": 8,
    "xlm_roberta_base_forward_gen": 8,
    "xlm_roberta_base_train_gen": 2,
    "xlm_roberta_large_it_mnli_forward_gen": 4,
    "xlm_roberta_large_ner_hrl_forward_gen": 8,
    "xlm_roberta_large_qa_multilingual_finedtuned_ru_forward_gen": 8,
    "xlm_roberta_large_squad2_forward_gen": 8,
    "xlm_roberta_large_xnli_forward_gen": 8,
    "yolos_small_forward_gen": 8,
    "zeroaraelectra_forward_gen": 16,
    # go/keep-sorted end
}
# pyformat: enable


_GEN_MODEL_CONFIGS: dict[str, ExtraConfigs] = {
    k: ExtraConfigs(
        **_GEN_MODEL_CONFIGS_RAW.get(k, {}),
        full_scope_batch_size=_GEN_FULL_SCOPE_BATCH_SIZES.get(k, 1),
    )
    for k in _GEN_MODEL_CONFIGS_RAW.keys() | _GEN_FULL_SCOPE_BATCH_SIZES.keys()
}
