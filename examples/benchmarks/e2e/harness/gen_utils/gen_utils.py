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
from examples.benchmarks.e2e.harness import context as context_lib
from examples.benchmarks.e2e.harness.gen_utils import load_model

ModelLoadFn = Callable[
    [context_lib.Context, str, bool], tuple[torch.nn.Module, dict[str, Any]]
]


def _get_models_csv_file():
  """Internalizes different ways to locate models.csv."""
  # 1. Try importlib (Standard for installed package/google3)
  try:
    csv_file = importlib.resources.files("torch_tpu").joinpath(
        "examples/benchmarks/e2e/harness/gen_utils/models.csv"
    )
    if csv_file.is_file():
      return csv_file
  except Exception:
    pass

  # 2. Try Bazel runfiles (For tests/dev in Bazel)
  runfiles_dir = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
  if runfiles_dir:
    path_main = (
        pathlib.Path(runfiles_dir)
        / "_main"
        / "examples/benchmarks/e2e/harness/gen_utils/models.csv"
    )
    if path_main.is_file():
      return path_main

    path_torch_tpu = (
        pathlib.Path(runfiles_dir)
        / "torch_tpu"
        / "examples/benchmarks/e2e/harness/gen_utils/models.csv"
    )
    if path_torch_tpu.is_file():
      return path_torch_tpu

  # 3. Try local path (Fallback)
  local_path = pathlib.Path(
      "examples/benchmarks/e2e/harness/gen_utils/models.csv"
  )
  if local_path.is_file():
    return local_path

  raise FileNotFoundError("models.csv not found.")


def dummy_compute_loss(model, input_args, input_kwargs) -> torch.Tensor:
  """A dummy compute loss function for models that do not return a loss naturally."""
  out = model(*input_args, **input_kwargs)
  if hasattr(out, "last_hidden_state"):
    return out.last_hidden_state.sum()
  if hasattr(out, "pooler_output"):
    return out.pooler_output.sum()
  if (
      isinstance(out, (tuple, list))
      and len(out) > 0
      and torch.is_tensor(out[0])
  ):
    return out[0].sum()
  if torch.is_tensor(out):
    return out.sum()
  if isinstance(out, dict):
    for v in out.values():
      if torch.is_tensor(v):
        return v.sum()
  raise TypeError(
      "dummy_compute_loss could not find a tensor to sum in"
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


# pyformat: disable
_GEN_MODEL_CONFIGS_RAW: dict[str, dict[str, Any]] = {
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
    "bart_large_cnn_train_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "bart_large_mnli_forward_gen": {
        "skipped_run_modes": {
            "eager_default",
            "eager_optimized",
        },
    },
    "bart_large_mnli_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_optimized",
        },
    },
    "bert_large_uncased_whole_word_masking_squad2_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "blip_image_captioning_base_train_gen": {
        "skipped_run_modes": {
            "eager_optimized",
        },
    },
    "blip_vqa_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
        },
    },
    "blip_vqa_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "blip_vqa_capfilt_large_forward_gen": {
        "skipped_run_modes": {
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
    "clap_htsat_unfused_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "colmodernvbert_merged_st_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
            "compiled",
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
    "csm_1b_forward_gen": {
        "skipped_run_modes": {
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
    "decision_transformer_gym_hopper_medium_forward_gen": {
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
    "deplot_forward_gen": {
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
    "distiluse_base_multilingual_cased_v2_forward_gen": {
        "skipped_run_modes": {
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
    "e5_omni_7b_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
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
    "gemma_4_e4b_it_forward_gen": {
        "skipped_run_modes": {
            "eager_optimized",
        },
    },
    "gemma_4_e4b_it_train_gen": {
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
    "granite_embedding_small_english_r2_forward_gen": {
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
    "hf_seamless_m4t_medium_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "higgs_tts_2_3b_base_forward_gen": {
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
    "inclusively_reformulation_it5_train_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "instructor_large_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "kosmos_2_patch14_224_forward_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "lahja_sa_ahmad_v1_forward_gen": {
        "skipped_run_modes": {
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
    "layoutlm_document_qa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "mbart_large_50_many_to_many_mmt_train_gen": {
        "skipped_run_modes": {
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
    "mdeberta_v3_base_xnli_multilingual_nli_2mil7_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
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
    "modernbert_base_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "modernbert_base_train_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "music_flamingo_2601_hf_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
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
    "owlv2_base_patch16_ensemble_forward_gen": {
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
    "owlvit_base_patch32_forward_gen": {
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
            "eager_default",
        },
    },
    "qwen2_audio_7b_instruct_forward_gen": {
        "skipped_run_modes": {
            "eager_optimized",
        },
    },
    "qwen2_audio_7b_instruct_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
        },
    },
    "qwen3_5_9b_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "rtdetr_r101vd_coco_o365_forward_gen": {
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
    "sam3_litetext_s0_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "sam_hq_vit_base_forward_gen": {
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
    "sam_vit_base_forward_gen": {
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
    "siglip_so400m_patch14_384_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
            "eager_default",
        },
    },
    "speecht5_vc_train_gen": {
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
    "tapas_base_finetuned_sqa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "tapas_base_finetuned_sqa_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "tapas_base_finetuned_wtq_forward_gen": {
        "skipped_run_modes": {
            "compiled",
        },
    },
    "tapas_base_finetuned_wtq_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "tapas_large_finetuned_sqa_forward_gen": {
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
    "text_summarization_train_gen": {
        "skipped_run_modes": {
            "eager_default",
        },
    },
    "time_anchor_modernbert_32m_forward_gen": {
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
    "timesfm_2_5_200m_transformers_forward_gen": {
        "skipped_run_modes": {
            "eager_default",
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
    "videomae_base_forward_gen": {
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
    "vilt_b32_finetuned_vqa_forward_gen": {
        "skipped_run_modes": {
            "compiled",
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
    "xlm_roberta_large_squad2_train_gen": {
        "skipped_run_modes": {
            "compiled",
            "eager_default",
            "eager_optimized",
        },
    },
    "xlm_roberta_large_xnli_train_gen": {
        "skipped_run_modes": {
            "compiled",
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
