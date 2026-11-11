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

"""Provides a unified registry for instantiating PyTorch models.

Supports models from various sources.

This module defines a generic interface for model providers (`BaseProvider`) and
implementations for popular model libraries:
    - Torchvision
    - TIMM (PyTorch Image Models)
    - Hugging Face Transformers

It exposes a `ModuleRegistry` that aggregates these providers, allowing users
to list available models and retrieve `ModuleSpec` objects containing the
instantiated model and factory functions for generating compatible sample
inputs.
"""

# pylint: disable=g-import-not-at-top

import abc
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
import contextlib
import copy
import dataclasses
import enum
import functools
from importlib import abc as importlib_abc
from importlib import machinery as importlib_machinery
from importlib import resources
from importlib import util as importlib_util
from importlib.resources import abc as resources_abc
import inspect
import json
import os
import pathlib
import sys
import tempfile
import types
from typing import Any, overload
import uuid

from absl import flags
from absl import logging
from etils import epath
from google.api_core import exceptions as gcp_exceptions
from google.api_core import retry as gcp_retry
from google.cloud import storage
from safetensors import torch as safetensors_torch
import torch

from tests import module_cache
from tests import transformers_inputs

try:
  import diffusers
  from diffusers.models import auto_model

  _HAS_DIFFUSERS = True
except ImportError:
  _HAS_DIFFUSERS = False

try:
  import timm

  _HAS_TIMM = True
except ImportError:
  _HAS_TIMM = False

try:
  import torchvision

  _HAS_TORCHVISION = True
except ImportError:
  _HAS_TORCHVISION = False

try:
  import transformers

  _HAS_TRANSFORMERS = True
except ImportError:
  _HAS_TRANSFORMERS = False

if "torchaudio" not in sys.modules and (
    importlib_util.find_spec("torchaudio") is None
):
  _torchaudio_stub = types.ModuleType("torchaudio")
  setattr(_torchaudio_stub, "__version__", "2.1.0")
  _torchaudio_stub.__spec__ = importlib_machinery.ModuleSpec("torchaudio", None)
  _functional_stub = types.ModuleType("torchaudio.functional")
  _functional_stub.__spec__ = importlib_machinery.ModuleSpec(
      "torchaudio.functional", None
  )

  def _fake_resample(
      waveform: torch.Tensor,
      orig_freq: int | float,
      new_freq: int | float,
      *_unused_args: Any,
      **_unused_kwargs: Any,
  ) -> torch.Tensor:
    ratio = float(new_freq) / float(orig_freq)
    new_len = max(1, int(waveform.shape[-1] * ratio))
    if waveform.is_meta:
      return torch.empty(
          (*waveform.shape[:-1], new_len),
          device=waveform.device,
          dtype=waveform.dtype,
      )
    orig_shape = waveform.shape
    w = waveform.reshape(-1, 1, orig_shape[-1])
    out = torch.nn.functional.interpolate(
        w, size=new_len, mode="linear", align_corners=False
    )
    return out.reshape(*orig_shape[:-1], new_len)

  _transforms_stub = types.ModuleType("torchaudio.transforms")
  _transforms_stub.__spec__ = importlib_machinery.ModuleSpec(
      "torchaudio.transforms", None
  )

  class _DummyTransform(torch.nn.Module):

    def __init__(self, *args: Any, **kwargs: Any):
      super().__init__()

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
      return x

  setattr(_transforms_stub, "Spectrogram", _DummyTransform)
  setattr(_transforms_stub, "MelSpectrogram", _DummyTransform)
  setattr(_transforms_stub, "Resample", _DummyTransform)
  setattr(
      _functional_stub,
      "compute_deltas",
      lambda spec, *a, **k: torch.zeros_like(spec),
  )
  setattr(
      _torchaudio_stub, "load", lambda *a, **k: (torch.zeros(1, 16000), 16000)
  )
  setattr(_torchaudio_stub, "save", lambda *a, **k: None)
  setattr(_torchaudio_stub, "transforms", _transforms_stub)
  setattr(_functional_stub, "resample", _fake_resample)
  setattr(_torchaudio_stub, "functional", _functional_stub)
  sys.modules["torchaudio"] = _torchaudio_stub
  sys.modules["torchaudio.functional"] = _functional_stub
  sys.modules["torchaudio.transforms"] = _transforms_stub

  class _TorchaudioFinder(importlib_abc.MetaPathFinder):

    def find_spec(self, fullname, path, target=None):
      if fullname == "torchaudio" or fullname.startswith("torchaudio."):
        return importlib_machinery.ModuleSpec(fullname, None)
      return None

  sys.meta_path.insert(0, _TorchaudioFinder())

  if _HAS_TRANSFORMERS and "transformers" in sys.modules:
    from transformers.utils import import_utils

    import_utils.is_torchaudio_available = lambda: True
    tr_mod = sys.modules.get("transformers")
    tr_utils = getattr(tr_mod, "utils", None)
    if tr_utils and hasattr(tr_utils, "is_torchaudio_available"):
      setattr(tr_utils, "is_torchaudio_available", lambda: True)
    if "torchaudio" in import_utils.BACKENDS_MAPPING:
      msg = import_utils.BACKENDS_MAPPING["torchaudio"][1]
      import_utils.BACKENDS_MAPPING["torchaudio"] = (lambda: True, msg)
    for mod in list(sys.modules.values()):
      if hasattr(mod, "_object_missing_backend") and isinstance(
          mod._object_missing_backend, dict
      ):
        keys_to_pop = [
            k
            for k, v in mod._object_missing_backend.items()
            if "torchaudio" in v
        ]
        for k in keys_to_pop:
          mod._object_missing_backend.pop(k, None)

_AUDIO_MODEL_TYPES = (
    # go/keep-sorted start
    "audio",
    "audio-flamingo",
    "audio-spectrogram-transformer",
    "audio_flamingo",
    "bark",
    "clap",
    "cohere_asr",
    "csm",
    "dac",
    "data2vec-audio",
    "dia",
    "encodec",
    "fastspeech",
    "fastspeech2",
    "gemma3naudio",
    "gemma4_audio",
    "gemma4audio",
    "granite-speech",
    "granite_speech",
    "higgs_audio_v2",
    "higgs_audio_v2_tokenizer",
    "hubert",
    "lasr_ctc",
    "mimi",
    "moonshine",
    "moonshine_streaming",
    "moshi",
    "musicgen",
    "omnitoken2wav",
    "parakeet",
    "parakeet_ctc",
    "parakeet_encoder",
    "parakeet_rnnt",
    "parakeet_tdt",
    "pe-a-frame",
    "pe-audio",
    "pe_audio",
    "pe_audio_encoder",
    "pop2piano",
    "qwen2-audio",
    "qwen2_5_omni_audio_encoder",
    "qwen2_audio",
    "qwen3_omni_moe_audio_encoder",
    "scail",
    "sew",
    "sew-d",
    "speech",
    "speecht5",
    "unispeech-sat",
    "univnet",
    "vibevoice_acoustic_tokenizer",
    "vibevoice_asr",
    "vits",
    "voxtral",
    "voxtral_realtime",
    "wav2vec2",
    "wav2vec2-conformer",
    "wavlm",
    "whisper",
    "xcodec",
    "xcodec2",
    # go/keep-sorted end
)
_VISION_MODEL_TYPES = (
    # go/keep-sorted start
    "beit",
    "bit",
    "clip",
    "convnext",
    "convnextv2",
    "cvt",
    "d_fine",
    "data2vec-vision",
    "deimv2",
    "deit",
    "depth",
    "depth_anything",
    "depth_pro",
    "detr",
    "dinat",
    "dino",
    "dinov2",
    "dinov2_with_registers",
    "dinov3",
    "dinov3_vit",
    "dpt",
    "edgetam_video",
    "efficientloftr",
    "efficientnet",
    "eomt",
    "focalnet",
    "gemma4_vision",
    "glpn",
    "hgnet_v2",
    "hiera",
    "ijepa",
    "lightglue",
    "manga-ocr",
    "mask2former",
    "maskformer",
    "mgp-str",
    "mobilenet",
    "mobilenetv2",
    "oneformer",
    "owlv2",
    "pixio",
    "poolformer",
    "pp_doclayout_v3",
    "pp_formulanet",
    "pp_ocrv5_mobile_rec",
    "pp_ocrv6_small_rec",
    "pvt",
    "pvt_v2",
    "qianfan_ocr",
    "regnet",
    "resnet",
    "rf_detr",
    "sam",
    "sam2",
    "sam2_video",
    "sam3_lite_text",
    "sam3_video",
    "sam_hq",
    "sapiens2",
    "segformer",
    "slanet",
    "superglue",
    "superpoint",
    "swiftformer",
    "swin",
    "swin2sr",
    "swinv2",
    "table-transformer",
    "textnet",
    "timesformer",
    "timm_backbone",
    "timm_wrapper",
    "trocr",
    "upernet",
    "videomae",
    "videoprism",
    "vision",
    "vit",
    "vitpose",
    "vjepa",
    "vjepa2",
    "yolos",
    "zoedepth",
    # go/keep-sorted end
)
_TEXT_MODEL_TYPES = (
    # go/keep-sorted start
    "albert",
    "autoformer",
    "bert",
    "big_bird",
    "blt",
    "bros",
    "camembert",
    "canine",
    "convbert",
    "data2vec-text",
    "deberta",
    "deberta-v2",
    "distilbert",
    "dpr",
    "electra",
    "esm",
    "fnet",
    "funnel",
    "informer",
    "jina_embeddings_v3",
    "layoutlm",
    "layoutlmv2",
    "layoutlmv3",
    "lilt",
    "longformer",
    "luke",
    "markuplm",
    "megatron-bert",
    "mobilebert",
    "modernbert",
    "mpnet",
    "mra",
    "nystromformer",
    "openai_privacy_filter",
    "patchtsmixer",
    "patchtst",
    "perceiver",
    "reformer",
    "roberta",
    "roformer",
    "splinter",
    "tapas",
    "time_series_transformer",
    "xlm",
    "xlm-prophetnet",
    "xlm-roberta",
    "xlnet",
    "xlstm",
    "xmod",
    "yoso",
    # go/keep-sorted end
)
_CAUSAL_LM_MODEL_TYPES = (
    # go/keep-sorted start
    "afmoe",
    "apertus",
    "arcee",
    "bamba",
    "biogpt",
    "bitnet",
    "bloom",
    "codegen",
    "cohere",
    "cohere2",
    "cpmant",
    "ctrl",
    "dbrx",
    "decision_transformer",
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
    "diffllama",
    "dots1",
    "ernie",
    "ernie4_5",
    "ernie4_5_moe",
    "exaone4",
    "falcon",
    "falcon_h1",
    "falcon_mamba",
    "flex_olmo",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma3_text",
    "gemma3n",
    "gemma3n_text",
    "gemma4",
    "gemma4_assistant",
    "gemma4_text",
    "gemma4_unified",
    "gemma4_unified_assistant",
    "gemma4_unified_text",
    "glm",
    "glm4",
    "glm4_moe",
    "gpt",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gpt_neox_japanese",
    "gpt_oss",
    "gptj",
    "granite",
    "granitemoe",
    "granitemoehybrid",
    "helium",
    "hrm_text",
    "hunyuan_v1_dense",
    "jais2",
    "jamba",
    "jetmoe",
    "kosmos_2_5_text_model",
    "lfm2",
    "lfm2_moe",
    "llama",
    "llama4_text",
    "mamba",
    "mamba2",
    "ministral",
    "ministral3",
    "mistral",
    "mistral3",
    "mistral4",
    "mixtral",
    "mllama_text_model",
    "modernbert-decoder",
    "mpt",
    "nanochat",
    "nemotron",
    "nemotron_h",
    "olmo",
    "olmo2",
    "olmo3",
    "olmo_hybrid",
    "openai-gpt",
    "opt",
    "perception_lm",
    "persimmon",
    "phi",
    "phi3",
    "phi4",
    "phimoe",
    "qwen",
    "qwen2",
    "qwen2_5",
    "qwen2_moe",
    "qwen3",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_5_text",
    "qwen3_moe",
    "recurrent_gemma",
    "rwkv",
    "smollm3",
    "solar_open",
    "stablelm",
    "starcoder2",
    "timesfm2_5",
    "vaultgemma",
    "xglm",
    "xlstm",
    "youtu",
    "zamba",
    "zamba2",
    # go/keep-sorted end
)
_SEQ2SEQ_MODEL_TYPES = (
    # go/keep-sorted start
    "bart",
    "blenderbot",
    "blenderbot-small",
    "encoder-decoder",
    "fsmt",
    "led",
    "m2m",
    "m2m_100",
    "marian",
    "mbart",
    "mt5",
    "mvp",
    "nllb",
    "pegasus",
    "prophetnet",
    "rag",
    "seamless_m4t",
    "seamless_m4t_v2",
    "switch_transformers",
    "t5",
    "whisper",
    # go/keep-sorted end
)
_VISION_LANGUAGE_MODEL_TYPES = (
    # go/keep-sorted start
    "align",
    "audio-flamingo",
    "audio_flamingo",
    "audioflamingo",
    "blip",
    "blip-2",
    "blip_2",
    "bridgetower",
    "chameleon",
    "chameleon_vqgan",
    "chmv2",
    "clip",
    "clvp",
    "colpali",
    "cosmos3",
    "donut-swin",
    "emu3",
    "fast_vlm",
    "flava",
    "florence2",
    "fuyu",
    "git",
    "glm4v",
    "glm_ocr",
    "got_ocr2",
    "grounding-dino",
    "grounding_dino",
    "groundingdino",
    "groupvit",
    "holo",
    "hunyuan_vl",
    "idefics",
    "idefics2",
    "idefics3",
    "instructblip",
    "internvl",
    "janus",
    "kosmos-2",
    "kosmos-2.5",
    "lighton_ocr",
    "llav",
    "llava",
    "llava_next",
    "llava_next_video",
    "llmdet",
    "llmdet-swin",
    "llmdet_swin",
    "lxmert",
    "mimo_v2_flash",
    "minicpmv4_6",
    "mllama",
    "musicflamingo",
    "oneformer",
    "ovis2",
    "owlv2",
    "owlvit",
    "paligemma",
    "pix2struct",
    "pp_formulanet",
    "qformer",
    "qwen2-audio",
    "qwen2_5_omni",
    "qwen2_5_vl",
    "qwen2_audio",
    "qwen2_vl",
    "qwen3_omni_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "sam3",
    "sam3_lite_text",
    "shieldgemma",
    "shieldgemma2",
    "siglip",
    "smolvlm",
    "tvp",
    "udop",
    "videoprism",
    "vilt",
    "vision-text-dual-encoder",
    "vision_text_dual_encoder",
    "xclip",
    # go/keep-sorted end
)
# Multimodal is the union of all multimodal subtypes
_MULTIMODAL_MODEL_TYPES = _VISION_LANGUAGE_MODEL_TYPES
_ALL_SUPPORTED_TRANSFORMERS_MODEL_TYPES = (
    _AUDIO_MODEL_TYPES
    + _VISION_MODEL_TYPES
    + _CAUSAL_LM_MODEL_TYPES
    + _SEQ2SEQ_MODEL_TYPES
    + _MULTIMODAL_MODEL_TYPES
    + _TEXT_MODEL_TYPES
)

_MAX_SEQ_LEN_HEURISTIC_CAP = 1_000_000

_WEIGHTS_BASE_PATH = flags.DEFINE_string(
    "weights_base_path",
# "$DATA_ROOT/$WEIGHTS_SUBDIR"
    "",
    "Default base location of model configs and weights.",
)

_GCS_WEIGHTS_BUCKET = flags.DEFINE_string(
    "gcs_weights_bucket",
    "torchtpu-test",
    "GCS bucket name for fetching model configs and weights in OSS.",
)

_GCS_WEIGHTS_PREFIX = flags.DEFINE_string(
    "gcs_weights_prefix",
    "weights",
    "Prefix within the GCS bucket where weights and configs are stored.",
)

# Per-request HTTP timeout for GCS downloads, so an unreachable GCS endpoint
# fails fast and falls back instead of stalling the caller on the library's
# default 60s timeout.
_GCS_DOWNLOAD_TIMEOUT_SECONDS = 15

_RETRY_TRANSIENT_IO = gcp_retry.Retry(
    predicate=lambda exc: isinstance(exc, OSError)
    and not isinstance(exc, (FileNotFoundError, NotADirectoryError)),
    initial=0.5,
    maximum=2.0,
    timeout=5.0,
)


def _download_gcs_blob(
    bucket_name: str,
    blob_name: str,
    dest_path: pathlib.Path,
) -> bool:
  """Downloads a blob from Google Cloud Storage using google.cloud.storage.

  Args:
    bucket_name: Name of the GCS bucket (e.g. 'torchtpu-test').
    blob_name: Path of the object in the bucket (e.g.
      'weights/transformers/google/gemma-2-2b/config.json').
    dest_path: Local pathlib.Path destination.

  Returns:
    True if download succeeded, False otherwise.
  """
  dest_path.parent.mkdir(parents=True, exist_ok=True)
  temp_dest = dest_path.with_name(
      f"{dest_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
  )
  try:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(
        str(temp_dest), timeout=_GCS_DOWNLOAD_TIMEOUT_SECONDS
    )
    temp_dest.replace(dest_path)
    logging.info(
        "Successfully downloaded gs://%s/%s to %s",
        bucket_name,
        blob_name,
        dest_path,
    )
    return True
  except gcp_exceptions.NotFound:
    logging.info("GCS blob gs://%s/%s does not exist.", bucket_name, blob_name)
    return False
  except Exception as exc:  # pylint: disable=broad-except
    logging.warning(
        "Failed to download gs://%s/%s: %s", bucket_name, blob_name, exc
    )
    return False
  finally:
    try:
      temp_dest.unlink(missing_ok=True)
    except OSError:
      pass


_PROVIDER_ALIASES: dict[str, str] = {
    "sentence-transformers": "transformers",
}


class _ConfigWrapper:
  """Wrapper around HF config objects for attribute and dict access."""

  def __init__(self, data: Any):
    if isinstance(data, _ConfigWrapper):
      object.__setattr__(self, "_data", data._data)
    else:
      object.__setattr__(self, "_data", data)

  def unwrap(self) -> Any:
    """Returns the underlying wrapped config object or dictionary."""
    return self._data

  def __copy__(self) -> "_ConfigWrapper":
    return _ConfigWrapper(copy.copy(self._data))

  def __deepcopy__(self, memo: dict[int, Any]) -> "_ConfigWrapper":
    return _ConfigWrapper(copy.deepcopy(self._data, memo))

  def __getstate__(self) -> dict[str, Any]:
    return {"_data": self._data}

  def __setstate__(self, state: Mapping[str, Any]) -> None:
    object.__setattr__(self, "_data", state["_data"])

  def __getattr__(self, name: str) -> Any:
    if name == "_data" or "_data" not in self.__dict__:
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    data = self._data
    if data is None:
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    if getattr(data, "allow_global_per_layer_attribute_access", None) is False:
      try:
        data.allow_global_per_layer_attribute_access = True
      except (AttributeError, TypeError):
        pass
    if isinstance(data, (dict, Mapping)):
      if name in data:
        return _wrap_config(data[name])
      raise AttributeError(
          f"'{type(self).__name__}' object has no attribute '{name}'"
      )
    if hasattr(data, "_mock_name") and name not in getattr(
        data, "__dict__", {}
    ):
      raise AttributeError(
          f"'{type(self).__name__}' mock has no attribute '{name}'"
      )
    val = getattr(data, name)
    return _wrap_config(val)

  def __setattr__(self, name: str, value: Any) -> None:
    if name == "_data":
      object.__setattr__(self, name, value)
      return
    if isinstance(self._data, MutableMapping):
      self._data[name] = value
    elif hasattr(self._data, "__dict__"):
      self._data.__dict__[name] = value
    else:
      setattr(self._data, name, value)

  def get(self, key: str, default: Any = None) -> Any:
    data = self._data
    if getattr(data, "allow_global_per_layer_attribute_access", None) is False:
      try:
        data.allow_global_per_layer_attribute_access = True
      except (AttributeError, TypeError):
        pass
    if isinstance(data, (dict, Mapping)):
      val = data.get(key, default)
    elif hasattr(data, "_mock_name") and key not in getattr(
        data, "__dict__", {}
    ):
      val = default
    else:
      try:
        val = getattr(data, key, default)
      except (AttributeError, TypeError):
        val = default
    if val is default:
      return default
    if (
        val is not None
        and default is not None
        and not isinstance(val, type(default))
    ):
      try:
        val = type(default)(val)
      except (ValueError, TypeError):
        pass
    return _wrap_config(val)

  def __getitem__(self, key: Any) -> Any:
    data = self._data
    if isinstance(data, (dict, Mapping)) or hasattr(data, "__getitem__"):
      return _wrap_config(data[key])
    if isinstance(key, str):
      try:
        return getattr(self, key)
      except AttributeError as exc:
        raise KeyError(key) from exc
    raise KeyError(key)

  def __contains__(self, key: str) -> bool:
    if isinstance(self._data, (dict, Mapping)):
      return key in self._data
    if hasattr(self._data, "_mock_name") and key not in getattr(
        self._data, "__dict__", {}
    ):
      return False
    return hasattr(self._data, key)

  def __bool__(self) -> bool:
    return bool(self._data)

  def __len__(self) -> int:
    if isinstance(self._data, (dict, Mapping)) or hasattr(
        self._data, "__len__"
    ):
      return len(self._data)
    return len(dir(self._data))

  def __iter__(self):
    if isinstance(self._data, (dict, Mapping)) or hasattr(
        self._data, "__iter__"
    ):
      return iter(self._data)
    return iter(dir(self._data))

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if callable(self._data):
      return _wrap_config(self._data(*args, **kwargs))
    raise TypeError(f"'{type(self).__name__}' object is not callable")

  def __str__(self) -> str:
    return str(self._data)

  def __repr__(self) -> str:
    return f"{type(self).__name__}({self._data!r})"


def _wrap_config(config: Any) -> Any:
  """Wraps a config in _ConfigWrapper if not already wrapped."""
  if config is None or isinstance(config, _ConfigWrapper):
    return config
  if isinstance(
      config,
      (
          int,
          float,
          str,
          bool,
          bytes,
          list,
          tuple,
          set,
          frozenset,
          type,
          types.FunctionType,
          types.BuiltinFunctionType,
          types.MethodType,
          torch.dtype,
          torch.device,
          torch.Tensor,
      ),
  ):
    return config
  return _ConfigWrapper(config)


def is_supported_transformers_model_type(
    model_type: str, config: Any | None = None
) -> bool:
  """Returns True if model_type or config is supported by module_registry."""
  if not model_type and config is None:
    return False
  if model_type and any(
      k in model_type.lower() for k in _ALL_SUPPORTED_TRANSFORMERS_MODEL_TYPES
  ):
    return True
  if config is not None:
    cfg = _wrap_config(config)
    archs = cfg.get("architectures", [])
    for arch in archs:
      arch_lower = arch.lower()
      if (
          any(k in arch_lower for k in _ALL_SUPPORTED_TRANSFORMERS_MODEL_TYPES)
          or "causallm" in arch_lower
          or "lmheadmodel" in arch_lower
      ):
        return True
  return False


class Modality(enum.Enum):
  AUDIO = "audio"
  CAUSAL_LM = "causal_lm"
  DIFFUSION = "diffusion"
  MULTIMODAL = "multimodal"
  SEQ2SEQ = "seq2seq"
  TEXT = "text"
  UNKNOWN = "unknown"
  VISION = "vision"


ModuleFactory = module_cache.ModuleFactory


class ModuleSpec:
  """A specification container for a PyTorch model and its inputs.

  Attributes:
    module_factory: A `ModuleFactory` that returns an instantiated
      `torch.nn.Module`. It accepts an optional `dtype` (`torch.dtype`).
      Transformers and Diffusers models are constructed in it, while Torchvision
      and TIMM models are constructed in the default dtype and cast to it. If
      `dtype` is None, the provider chooses the dtype (e.g. the config dtype of
      a Transformers model).
    sample_inputs_factory: A callable that generates compatible input tensors.
      It accepts optional `shape` (Sequence) and `device` (str) arguments and
      returns a tuple containing `(args, kwargs)` for the model's forward pass.
    preprocessor_factory: A callable that generates a preprocessor for models
      that need it. None by default.
    config: Optional configuration object associated with the model (e.g., a
      Transformers `AutoConfig`).
    modality: the modality of the model, which specifies its type (e.g.
      causal_lm, vision, etc.).
  """

  def __init__(
      self,
      module_factory: ModuleFactory,
      sample_inputs_factory: Callable[
          [Sequence[int] | None, str | None],
          tuple[tuple[Any, ...], dict[str, Any]],
      ],
      preprocessor_factory: Callable[[], Any] | None = None,
      config: Any | None = None,
      modality: Modality = Modality.UNKNOWN,
  ):
    self.module_factory = module_factory
    self.sample_inputs_factory = sample_inputs_factory
    self.preprocessor_factory: Callable[[], Any] | None = (
        functools.cache(preprocessor_factory)
        if preprocessor_factory is not None
        else None
    )
    self.config = config
    self.modality = modality


def _cast_after_construction(
    create_module: Callable[[], torch.nn.Module],
) -> ModuleFactory:
  """Returns a module factory that casts the constructed model to `dtype`.

  Unlike Transformers and Diffusers models, Torchvision and TIMM models are
  constructed in the default dtype and cast to `dtype` afterwards. Their
  constructors take no dtype, and constructing them under another default dtype
  would also change values computed at construction that are not parameters
  (e.g. TIMM's stochastic depth rates from `torch.linspace`). The models are
  small, so the float32 copy before the cast costs little memory.

  Args:
    create_module: Constructs the model in the default dtype.

  Returns:
    The module factory.
  """

  def module_factory(dtype: torch.dtype | None = None) -> torch.nn.Module:
    model = create_module()
    return model if dtype is None else model.to(dtype)

  return module_factory


class BaseProvider(abc.ABC):
  """Abstract base class for model source providers."""

  def __init__(self, base_path: str | None = None, subdir: str | None = None):
    self._subdir = subdir

    if base_path is None:
      base_path = _WEIGHTS_BASE_PATH.value

    # If base_path is empty, assume no local cache is available
    self.has_cache_dir = bool(base_path)

    if self.has_cache_dir:
      self._base_path = epath.Path(base_path)
      if subdir:
        self._base_path = self._base_path / subdir
      if not _RETRY_TRANSIENT_IO(self._base_path.exists)():
        raise FileNotFoundError(f"Base path does not exist: {self._base_path}")
    else:
      self._base_path = None

  @property
  def _cloud_bucket_path(self) -> str | None:
    """Returns the GCS blob prefix, or None if GCS is not configured."""
    bucket = _GCS_WEIGHTS_BUCKET.value
    if not bucket:
      return None
    prefix = (
        _GCS_WEIGHTS_PREFIX.value.strip("/")
        if _GCS_WEIGHTS_PREFIX.value
        else ""
    )
    parts = [p for p in (prefix, self._subdir) if p]
    return "/".join(parts)

  def fetch_gcs_file(
      self, relative_path: str | pathlib.Path
  ) -> pathlib.Path | None:
    """Fetches a file from GCS, using local cache if already downloaded.

    Args:
      relative_path: Path relative to this provider's `_cloud_bucket_path`. For
        example, `"google/gemma-2-2b/config.json"`.

    Returns:
      The pathlib.Path to the cached/downloaded file, or None if GCS is not
      configured or the download failed.
    """
    bucket = _GCS_WEIGHTS_BUCKET.value
    cloud_path = self._cloud_bucket_path
    if not bucket or cloud_path is None:
      return None

    relative_path = pathlib.Path(relative_path)

    # Construct GCS blob name
    blob_name = (
        f"{cloud_path}/{relative_path}" if cloud_path else str(relative_path)
    )

    # Local destination in temp dir
    cache_dir = pathlib.Path(tempfile.gettempdir()) / "torch_tpu_cache"
    if self._subdir:
      cache_dir = cache_dir / self._subdir
    dest_path = cache_dir / relative_path

    if dest_path.exists():
      return dest_path

    if _download_gcs_blob(bucket, blob_name, dest_path):
      return dest_path

    return None

  @abc.abstractmethod
  def list_modules(self) -> Sequence[str]:
    """Lists the names of all models available from this provider.

    Returns:
      A list of model name strings.
    """
    pass

  @abc.abstractmethod
  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Retrieves the specification for a specific model.

    Args:
      name: The name of the model to retrieve.
      load_weights: If True, loads pre-trained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable to modify the model configuration. The
        config object is specific to the provider.
      **kwargs: Additional provider-specific arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    pass


def _create_torchvision_model(name: str) -> torch.nn.Module:
  """Deterministically instantiates a Torchvision model without weights."""
  tv_api = getattr(getattr(torchvision, "models", None), "_api", None)
  builtin_models = getattr(tv_api, "BUILTIN_MODELS", {})
  model_fn = (
      builtin_models.get(name) or builtin_models.get(name.lower())
      if isinstance(builtin_models, Mapping)
      else None
  )
  if callable(model_fn):
    params = inspect.signature(model_fn).parameters
    if "weights_backbone" in params:
      return torchvision.models.get_model(
          name, weights=None, weights_backbone=None
      )
  return torchvision.models.get_model(name, weights=None)


class TorchvisionProvider(BaseProvider):
  """Provider for standard Torchvision models."""

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="torchvision")

  def list_modules(self) -> Sequence[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/torchvision/{model_name}/weights.pt

    Returns:
      A list of model name strings.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    return [path.name for path in self._base_path.iterdir() if path.is_dir()]

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    if modify_config_hook is not None:
      raise ValueError("modify_config_hook is not supported for torchvision.")
    if load_weights and not self.has_cache_dir:
      raise ValueError(
          f"load_weights cannot be set to True for {name} when no cache"
          " directory is available."
      )
    if any(
        k in name.lower() for k in ("mvit", "r3d", "mc3", "r2plus1d", "s3d")
    ):
      default_shape = (1, 3, 16, 224, 224)
    else:
      default_shape = (1, 3, 224, 224)

    def _torchvision_input_factory(shape=None, device="cpu"):
      g = torch.Generator(device="cpu").manual_seed(42)
      return (
          (
              torch.randn(
                  shape if shape is not None else default_shape, generator=g
              ).to(device),
          ),
          {},
      )

    def _create_module():
      if load_weights and self._base_path:
        local_checkpoint = None
        local_dir = self._base_path / name

        if local_dir.exists() and local_dir.is_dir():
          for p in local_dir.iterdir():
            if p.suffix in (".pt", ".pth", ".bin", ".safetensors"):
              local_checkpoint = p
              break

        if local_checkpoint and local_checkpoint.exists():
          model = _create_torchvision_model(name)

          with local_checkpoint.open("rb") as f:
            if local_checkpoint.suffix == ".safetensors":
              state_dict = safetensors_torch.load(f.read())
            else:
              state_dict = torch.load(f, map_location="cpu")
          model.load_state_dict(state_dict)
          return model
        else:
          raise ValueError(
              f"Cannot load weights for {name} because checkpoint is missing"
              f" at {self._base_path / name}."
          )

      # If weights are not loaded, instantiate model with random weights
      return _create_torchvision_model(name)

    return ModuleSpec(
        _cast_after_construction(_create_module),
        _torchvision_input_factory,
        modality=Modality.VISION,
    )


class TimmProvider(BaseProvider):
  """Provider for TIMM (PyTorch Image Models)."""

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="timm")

  def list_modules(self) -> Sequence[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/timm/{model_name}/weights.pth

    Returns:
      A list of model name strings.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    return [path.name for path in self._base_path.iterdir() if path.is_dir()]

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a TIMM model.

    Attempts to determine the default input shape from the model's pretrained
    configuration. Defaults to (3, 224, 224) if configuration is unavailable.

    Args:
      name: Name (str) of the timm model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable that accepts and returns a
        timm.models.PretrainedCfg object to modify the model configuration.
      **kwargs: Additional keyword arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    if load_weights and not self.has_cache_dir:
      raise ValueError(
          f"load_weights cannot be set to True for {name} when no cache"
          " directory is available."
      )

    timm_name = (
        f"hf-hub:{name}"
        if ("/" in name and not name.startswith(("hf-hub:", "local-dir:")))
        else name
    )

    if timm.models.is_model(timm_name) and (
        "." not in timm_name
        or timm_name in timm.models.get_arch_pretrained_cfgs(timm_name)
    ):
      config = timm.models.get_pretrained_cfg(timm_name)
    else:
      config = None
      logging.warning(
          "Couldn't find config for %s.",
          timm_name,
      )

    if config and modify_config_hook is not None:
      config = modify_config_hook(config)

    def _create_module():
      if load_weights and self._base_path:
        local_checkpoint = self._base_path / name / f"{name}.pth"
        safetensors_checkpoint = self._base_path / name / "model.safetensors"
        bin_checkpoint = self._base_path / name / "pytorch_model.bin"

        if local_checkpoint.exists():
          with local_checkpoint.open("rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        elif safetensors_checkpoint.exists():
          with safetensors_checkpoint.open("rb") as f:
            state_dict = safetensors_torch.load(f.read())
        elif bin_checkpoint.exists():
          with bin_checkpoint.open("rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        else:
          raise ValueError(
              f"Cannot load weights for {name} because checkpoint is missing"
              f" at {local_checkpoint}, {safetensors_checkpoint}, or"
              f" {bin_checkpoint}."
          )

        model = timm.create_model(timm_name, pretrained=False)
        model.load_state_dict(state_dict)
        return model

      return timm.create_model(timm_name, pretrained=False)

    def _input_factory(shape=None, device="cpu"):
      g = torch.Generator(device="cpu").manual_seed(42)
      input_size = config.input_size if config else (3, 224, 224)
      final_shape = shape if shape else (1, *input_size)
      return ((torch.randn(final_shape, generator=g).to(device),), {})

    def _preprocessor_factory():
      if config:
        data_config = timm.data.resolve_data_config(
            {}, pretrained_cfg=config.to_dict()
        )
        transform = timm.data.create_transform(**data_config)
        return lambda img: transform(img).unsqueeze(0)
      return None

    return ModuleSpec(
        _cast_after_construction(_create_module),
        _input_factory,
        _preprocessor_factory,
        config,
        modality=Modality.VISION,
    )


@overload
def _safe_int(
    val: Any, default: None = None, min_val: int | None = None
) -> int | None:
  ...


@overload
def _safe_int(val: Any, default: int = 0, min_val: int | None = None) -> int:
  ...


def _safe_int(
    val: Any, default: int | None = 0, min_val: int | None = None
) -> int | None:
  """Safely converts a value to int, defaulting if invalid or a boolean."""
  if (
      val is None
      or isinstance(val, bool)
      or not isinstance(val, (int, float, str))
  ):
    return default
  try:
    res = int(val)
    if min_val is not None and res < min_val:
      return default
    return res
  except (ValueError, TypeError):
    return default


def _get_max_seq_len(config: Any, default: int = 512) -> int:
  """Heuristically determines the maximum sequence length from a config.

  Checks multiple attributes (e.g., `max_position_embeddings`, `n_positions`)
  and applies a cap to avoid unreasonably large allocations (e.g., for models
  using relative embeddings or integer-limit defaults).

  Args:
    config: The Transformers configuration object.
    default: The fallback sequence length if no attribute is found.

  Returns:
    The determined maximum sequence length.
  """
  cfg = _wrap_config(config)
  configs_to_check = [cfg]
  for sub in ("text_config", "encoder", "decoder"):
    sub_cfg = cfg.get(sub)
    if sub_cfg is not None:
      configs_to_check.append(_wrap_config(sub_cfg))

  found_vals = []
  for c in configs_to_check:
    for attr in (
        "max_text_len",
        "max_position_embeddings",
        "max_target_positions",
        "n_positions",
        "seq_length",
        "max_seq_len",
        "model_max_length",
    ):
      raw_val = c.get(attr)
      val = _safe_int(raw_val, default=0, min_val=1)
      if 0 < val < _MAX_SEQ_LEN_HEURISTIC_CAP:
        # Several HuggingFace encoder and seq2seq models offset their learned
        # positional embedding indices by `pad_token_id + 1` (computing
        # `position_ids = pad_token_id + 1 + arange(seq_len)` so indices
        # `0..pad_token_id` remain reserved for padding). This includes RoBERTa
        # (`FacebookAI/roberta-base`, `roberta-large`), XLM-RoBERTa
        # (`FacebookAI/xlm-roberta-base`, `xlm-roberta-large`), CamemBERT
        # (`almanach/camembert-base`), MPNet (`microsoft/mpnet-base`),
        # Longformer (`allenai/longformer-base-4096`), Data2VecText
        # (`facebook/data2vec-text-base`), and FSMT (`facebook/wmt19-ru-en`,
        # which uses `max_target_positions`). When `pad_token_id > 0` (typically
        # `1`), a sequence of length `val` produces a maximum position ID of
        # `val + pad_token_id`, which exceeds `nn.Embedding(val, hidden_size)`
        # and triggers a CUDA device-side out-of-bounds assertion. Subtracting
        # `pad_token_id + 1` keeps all generated position IDs within `[0, val)`.
        if attr in ("max_position_embeddings", "max_target_positions"):
          pad_id = _safe_int(
              c.get("pad_token_id") or cfg.get("pad_token_id"),
              default=0,
              min_val=0,
          )
          if pad_id > 0 and val > pad_id + 1:
            val -= pad_id + 1
        found_vals.append(val)
        break

  if found_vals:
    return min(found_vals)

  # Fallback for relative position models (T5, etc.) or missing data.
  return default


def _walk_package_resources(
    traversable: resources_abc.Traversable,
) -> Iterator[resources_abc.Traversable]:
  """Recursively yields Traversable objects for all files in a package."""
  for path in traversable.iterdir():
    if path.is_file():
      yield path
    elif path.is_dir():
      # Recursively call the function for subdirectories
      yield from _walk_package_resources(path)


def _is_diffusion(
    model_type: str, class_name: str, arch_name: str, model_name: str
) -> bool:
  return (
      any(
          k in model_type
          for k in (
              "diffusion",
              "unet",
              "ldm",
              "flux",
              "sdxl",
              "stable-diffusion",
          )
      )
      or any(
          k in class_name
          for k in (
              "unet",
              "ldm",
              "if-",
              "minit2i",
              "stablediffusion",
              "ddpm",
              "diffusion",
          )
      )
      or any(k in arch_name for k in ("unet", "diffusion"))
      or any(k in model_name for k in ("diffusion", "ldm", "if-", "minit2i"))
  )


def _is_multimodal(
    model_type: str,
    arch_name: str,
    has_text_cfg: bool,
    has_vision_cfg: bool,
    has_vocab_size: bool = False,
) -> bool:
  """Returns True if the model config represents a multimodal architecture."""
  # Open-vocabulary zero-shot models require text queries (input_ids) along
  # with images.
  if any(
      k in model_type
      for k in (
          "clipseg",
          "owlv2",
          "owlvit",
          "grounding_dino",
          "groundingdino",
          "grounding-dino",
      )
  ):
    return True
  if (
      any(
          k in arch_name
          for k in (
              "imageclassification",
              "imagesegmentation",
              "depthestimation",
              "objectdetection",
          )
      )
      and "shieldgemma" not in model_type
      and "shieldgemma" not in arch_name
  ):
    return False
  if (
      "audio_encoder" in model_type
      or "audioencoder" in model_type
      or model_type == "sam3_video"
  ):
    return False
  return (
      any(k in model_type for k in _MULTIMODAL_MODEL_TYPES)
      or "clip" in arch_name
      or "llava" in arch_name
      or "paligemma" in arch_name
      or "blip" in arch_name
      or (has_vision_cfg and (has_text_cfg or has_vocab_size))
      or (
          not _is_audio(model_type, arch_name)
          and ("omni" in model_type or "omni" in arch_name)
      )
  )


def _is_audio(model_type: str, arch_name: str) -> bool:
  return (
      any(k in model_type for k in _AUDIO_MODEL_TYPES)
      or "audio" in arch_name
      or "speech" in arch_name
      or "whisper" in arch_name
      or "wav2vec2" in arch_name
      or "asr" in arch_name
      or "asr" in model_type
  )


def _is_vision(
    model_type: str, arch_name: str, has_image_size: bool, has_vocab_size: bool
) -> bool:
  return (
      any(k in model_type for k in _VISION_MODEL_TYPES)
      or "image" in arch_name
      or "vit" in arch_name
      or "pixel_values" in arch_name
      or "resnet" in arch_name
      or "dinov2" in arch_name
      or "convnext" in arch_name
      or "depth" in arch_name
      or (has_image_size and not has_vocab_size)
  )


def _is_causal(model_type: str, arch_name: str) -> bool:
  return (
      any(k in model_type for k in _CAUSAL_LM_MODEL_TYPES)
      or "causallm" in arch_name
      or "llama" in arch_name
      or "gpt" in arch_name
      or "qwen" in arch_name
      or "mistral" in arch_name
      or "phi" in arch_name
      or "bloom" in arch_name
  )


def _is_seq2seq(model_type: str, arch_name: str, is_enc_dec: bool) -> bool:
  return (
      any(k in model_type for k in _SEQ2SEQ_MODEL_TYPES)
      or is_enc_dec
      or "conditionalgeneration" in arch_name
      or "t5" in arch_name
      or "bart" in arch_name
      or "marian" in arch_name
      or "pegasus" in arch_name
  )


def _is_text(model_type: str, arch_name: str) -> bool:
  return (
      any(k in model_type for k in _TEXT_MODEL_TYPES)
      or "bert" in arch_name
      or "encoder" in arch_name
  )


# Helper functions for TransformersProvider
def _determine_modality(config: Any) -> Modality:
  """Determines the modality of a Transformers model from its config.

  Args:
    config: The Transformers configuration object or dictionary.

  Returns:
    A Modality enum value.
  """
  class_name = ""
  model_name = ""
  archs = []
  model_type = ""
  is_enc_dec = False

  if config is not None:
    cfg = _wrap_config(config)
    class_name = cfg.get("_class_name", "").lower()
    model_name = cfg.get("_name_or_path", "").lower()
    model_type = cfg.get("model_type", "").lower()
    archs = cfg.get("architectures", [])
    is_enc_dec = cfg.get("is_encoder_decoder", False)
    has_text_cfg = cfg.get("text_config") is not None
    has_vision_cfg = cfg.get("vision_config") is not None
    backbone_cfg = cfg.get("backbone_config")
    has_image_size = (
        cfg.get("image_size") is not None
        or cfg.get("num_channels") is not None
        or (
            backbone_cfg is not None
            and backbone_cfg.get("image_size") is not None
        )
    )
    has_vocab_size = cfg.get("vocab_size") is not None
  else:
    has_text_cfg = False
    has_vision_cfg = False
    has_image_size = False
    has_vocab_size = False

  arch_name = archs[0].lower() if archs else ""
  base_name = model_name.lower().split("/")[-1]
  if (
      arch_name.endswith(("textmodel", "text_model"))
      or base_name.endswith(("-text", "_text"))
  ) and not has_vision_cfg:
    return Modality.TEXT

  if (
      arch_name.endswith(("visionmodel", "vision_model"))
      or model_type.endswith(("_vision_model", "_vision"))
      or base_name.endswith(("-vision", "_vision"))
  ) and not has_text_cfg:
    return Modality.VISION

  if _is_diffusion(model_type, class_name, arch_name, model_name):
    return Modality.DIFFUSION
  if _is_multimodal(
      model_type,
      arch_name,
      has_text_cfg,
      has_vision_cfg,
      has_vocab_size,
  ):
    return Modality.MULTIMODAL
  if _is_audio(model_type, arch_name):
    return Modality.AUDIO
  if _is_vision(model_type, arch_name, has_image_size, has_vocab_size):
    return Modality.VISION
  if _is_causal(model_type, arch_name):
    return Modality.CAUSAL_LM
  if _is_seq2seq(model_type, arch_name, is_enc_dec):
    return Modality.SEQ2SEQ
  if _is_text(model_type, arch_name):
    return Modality.TEXT

  # If we can't figure it out then assume modality is `text` for a transformers
  # model.
  return Modality.TEXT


def _parse_image_size(config: Any, default_size: int = 224) -> int:
  """Extracts image_size as an int from config, vision_config, or encoder."""
  cfg = _wrap_config(config)
  if "depth_pro" in str(cfg.get("model_type", "")):
    patch_size = _safe_int(cfg.get("patch_size"), default=384, min_val=1)
    ratios = cfg.get("scaled_images_ratios")
    min_ratio = (
        min(r for r in ratios if r > 0)
        if isinstance(ratios, (list, tuple)) and any(r > 0 for r in ratios)
        else 0.25
    )
    return max(int(patch_size / min_ratio), 1536)
  if any(
      k in str(cfg.get("model_type", "")).lower() for k in ["d_fine", "dfine"]
  ):
    return 640
  if "rf_detr" in str(cfg.get("model_type", "")).lower():
    return 560

  detector_cfg = cfg.get("detector_config")
  vlm_cfg = cfg.get("vlm_config")
  vision_config = (
      cfg.get("vision_config")
      or (vlm_cfg.get("vision_config") if vlm_cfg else None)
      or (detector_cfg.get("vision_config") if detector_cfg else None)
      or cfg.get("encoder")
      or cfg.get("backbone_config")
  )
  if vision_config and vision_config.get("model_type") == "pixtral":
    return 128
  backbone_cfg = vision_config.get("backbone_config") if vision_config else None
  raw_val = (
      (
          vision_config.get("image_size")
          or vision_config.get("resolution")
          or (
              backbone_cfg.get("image_size") or backbone_cfg.get("resolution")
              if backbone_cfg is not None
              and (
                  isinstance(backbone_cfg, Mapping)
                  or hasattr(backbone_cfg, "get")
              )
              else None
          )
      )
      if vision_config
      else None
  )
  if isinstance(raw_val, (list, tuple)) and raw_val:
    raw_val = raw_val[0]
  val = _safe_int(raw_val, default=None, min_val=1)
  if val is None:
    raw_val = cfg.get("image_size") or cfg.get("resolution")
    if isinstance(raw_val, (list, tuple)) and raw_val:
      raw_val = raw_val[0]
    val = _safe_int(raw_val, default=None, min_val=1)
  if val is None:
    pretrained_cfg = cfg.get("pretrained_cfg") or (
        vision_config.get("pretrained_cfg") if vision_config else None
    )
    raw_val = cfg.get("input_size") or (
        pretrained_cfg.get("input_size") if pretrained_cfg else None
    )
    if isinstance(raw_val, (list, tuple)) and raw_val:
      val = _safe_int(raw_val[-1], default=None, min_val=1)
    else:
      val = _safe_int(raw_val, default=None, min_val=1)
  if val is None:
    val = default_size

  return val


def _get_num_channels(config: Any, default_channels: int = 3) -> int:
  """Extracts num_channels from config or vision_config or encoder."""
  cfg = _wrap_config(config)
  vlm_cfg = cfg.get("vlm_config")
  vision_config = (
      cfg.get("vision_config")
      or (vlm_cfg.get("vision_config") if vlm_cfg else None)
      or cfg.get("encoder")
      or cfg.get("backbone_config")
  )
  val = _safe_int(
      vision_config.get("num_channels") if vision_config else None,
      default=None,
      min_val=1,
  )
  if val is None:
    val = _safe_int(
        cfg.get("num_channels"),
        default=default_channels,
        min_val=1,
    )
  return val


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype | None) -> Iterator[None]:
  """Sets the default floating point dtype within the context.

  Args:
    dtype: The dtype to construct floating point tensors in. None and
      non-floating point dtypes leave the default dtype unchanged.

  Yields:
    None.
  """
  if dtype is None or not dtype.is_floating_point:
    yield
    return
  original_dtype = torch.get_default_dtype()
  torch.set_default_dtype(dtype)
  try:
    yield
  finally:
    torch.set_default_dtype(original_dtype)


def _extract_target_dtype(config: Any) -> torch.dtype:
  """Extracts target torch_dtype from root or nested transformer configs."""
  cfg = _wrap_config(config)
  if cfg.get("model_type", "") == "gemma2":
    return torch.bfloat16

  dtype = cfg.get("torch_dtype") or cfg.get("dtype")

  if dtype is None:
    text_config = cfg.get("text_config")
    if text_config is not None:
      dtype = text_config.get("torch_dtype") or text_config.get("dtype")

  if dtype is None:
    vision_config = cfg.get("vision_config")
    if vision_config is not None:
      dtype = vision_config.get("torch_dtype") or vision_config.get("dtype")

  if dtype is None:
    thinker_config = cfg.get("thinker_config")
    if thinker_config is not None:
      dtype = thinker_config.get("torch_dtype") or thinker_config.get("dtype")
      if dtype is None:
        thinker_text = thinker_config.get("text_config")
        if thinker_text is not None:
          dtype = thinker_text.get("torch_dtype") or thinker_text.get("dtype")

  if isinstance(dtype, str):
    return getattr(torch, dtype, torch.float32)
  elif isinstance(dtype, torch.dtype):
    return dtype
  return torch.float32


def _load_transformers_preprocessor(
    model_dir_or_repo_id: epath.PathLike | str | None,
    modality: Modality | None = None,
) -> Any:
  """Stage 5: Deterministically loads and caches a model preprocessor.

  What this stage does:
    Inspects `model_dir_or_repo_id` for concrete processor configuration files
    (`preprocessor_config.json`, `processor_config.json`) and tokenizer files
    (`tokenizer_config.json`, `tokenizer.json`, `spiece.model`, `vocab.json`,
    `vocab.txt`, `merges.txt`), then invokes `AutoTokenizer.from_pretrained` or
    `AutoProcessor.from_pretrained` according to `modality` and caches the
    loaded instance in `module_cache.PREPROCESSOR_CACHE`.

  Why this is needed as a separate stage:
    Preprocessor loading is exposed lazily via `ModuleSpec.preprocessor_factory`
    and invoked on demand by input generators. Separating it from config and
    model instantiation avoids unnecessary disk I/O during `meta`-device or
    config-only tests, and inspecting which config files exist on disk removes
    speculative `AutoProcessor` -> `AutoTokenizer` exception-catching loops.

  Example:
    For `meta-llama/Llama-3.2-1B` (`Modality.CAUSAL_LM`), the directory contains
    `tokenizer_config.json` and `tokenizer.json` without a processor config;
    Stage 5 sees `prefer_tokenizer=True` and `has_tokenizer=True` and calls
    `AutoTokenizer.from_pretrained` directly. Conversely, for a vision-language
    model like `llava-hf/llava-1.5-7b-hf` (`Modality.MULTIMODAL`), the directory
    contains `preprocessor_config.json`, so Stage 5 sees
    `prefer_tokenizer=False`
    and `has_processor=True` and calls `AutoProcessor.from_pretrained`.

  Args:
    model_dir_or_repo_id: Local directory path or repo identifier.
    modality: The detected `Modality` of the model.

  Returns:
    The loaded HuggingFace tokenizer or processor, or None if unavailable.
  """
  if model_dir_or_repo_id is None or not _HAS_TRANSFORMERS:
    return None
  model_path_str = str(model_dir_or_repo_id)
  prefer_tokenizer = modality in (
      Modality.CAUSAL_LM,
      Modality.TEXT,
      Modality.SEQ2SEQ,
  )
  cache_key = ("transformers", model_path_str, prefer_tokenizer)

  def _loader() -> Any:
    exists = lambda p: _RETRY_TRANSIENT_IO(p.exists)()
    path = epath.Path(model_path_str)
    if exists(path) and _RETRY_TRANSIENT_IO(path.is_dir)():
      has_processor = exists(path / "preprocessor_config.json") or exists(
          path / "processor_config.json"
      )
      has_tokenizer = (
          exists(path / "tokenizer_config.json")
          or exists(path / "tokenizer.json")
          or exists(path / "spiece.model")
          or exists(path / "vocab.json")
          or exists(path / "vocab.txt")
          or exists(path / "merges.txt")
      )
      if prefer_tokenizer and has_tokenizer:
        load_fn = transformers.AutoTokenizer.from_pretrained
      elif not prefer_tokenizer and has_processor:
        load_fn = transformers.AutoProcessor.from_pretrained
      elif has_processor:
        load_fn = transformers.AutoProcessor.from_pretrained
      elif has_tokenizer:
        load_fn = transformers.AutoTokenizer.from_pretrained
      elif prefer_tokenizer:
        load_fn = transformers.AutoTokenizer.from_pretrained
      else:
        load_fn = transformers.AutoProcessor.from_pretrained
    elif prefer_tokenizer:
      load_fn = transformers.AutoTokenizer.from_pretrained
    else:
      load_fn = transformers.AutoProcessor.from_pretrained

    try:
      return load_fn(model_path_str)
    except Exception as exc:  # pylint: disable=broad-except
      logging.debug(
          "Could not load preprocessor from %s: %s", model_path_str, exc
      )
      return None

  return module_cache.PREPROCESSOR_CACHE.get_or_load(cache_key, _loader)


# go/keep-sorted
# Mapping of non-canonical or community model types to canonical transformers
# model types.
_MODEL_TYPE_ALIASES: dict[str, str] = {
    "LanguageBindAudio": "clip",
    "LanguageBindImage": "clip",
    "LanguageBindVideo": "clip",
    "aquila": "llama",
    "gemma3_text_only": "gemma3_text",
    "languagebind_audio": "clip",
    "languagebind_depth": "clip",
    "languagebind_thermal": "clip",
    "languagebind_video": "clip",
}

# go/keep-sorted
# Architecture class remapping for models with mismatched or legacy
# architecture names.
_ARCHITECTURE_ALIASES: dict[str, dict[str, str]] = {
    "gemma4_unified": {
        "Gemma4ForCausalLM": "Gemma4UnifiedForConditionalGeneration",
        "Gemma4ForConditionalGeneration": (
            "Gemma4UnifiedForConditionalGeneration"
        ),
    },
    "qwen2_5_omni": {
        "Qwen2_5OmniModel": "Qwen2_5OmniForConditionalGeneration",
    },
    "qwen2_5_vl_text": {
        "Qwen2_5_VLForConditionalGeneration": "Qwen2_5_VLTextModel",
        "Qwen2_5_VLModel": "Qwen2_5_VLTextModel",
    },
    "speech_to_text": {
        "AV2TextForConditionalGeneration": (
            "Speech2TextForConditionalGeneration"
        ),
    },
    "wav2vec2": {
        "HubertForSequenceClassification": "Wav2Vec2ForSequenceClassification",
    },
}

# go/keep-sorted
# Config attribute aliases to copy if missing on the config dictionary.
_CONFIG_KEY_ALIASES: dict[str, str] = {
    "num_nextn_predict_layers": "num_mtp_layers",
}

# go/keep-sorted
# Default values for attributes that are explicitly null in upstream configs.
_CONFIG_DEFAULTS_WHEN_NONE: dict[str, dict[str, Any]] = {
    "blip-2": {
        "image_token_index": 50265,
    },
    "blip_2": {
        "image_token_index": 50265,
    },
    "speech_to_text": {
        "scale_embedding": (
            True
        ),  # AV-HuBERT configs specify null, which breaks embedding scaling
    },
}

_MOE_TOPK_KEYS: tuple[str, ...] = (
    "num_experts_per_tok",
    "experts_per_token",
    "moe_topk",
    "num_selected_experts",
)

# go/keep-sorted
# Model types whose float parameters must be zero-initialized before forward.
_ZERO_INIT_WEIGHT_MODEL_TYPES: frozenset[str] = frozenset({
    "tapas",
    "vits",
})

# go/keep-sorted
# Candidate AutoModel classes tried in priority order by modality.
_CANDIDATE_AUTO_CLASSES_BY_MODALITY: dict[Modality, tuple[str, ...]] = {
    Modality.AUDIO: (
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForCTC",
        "AutoModelForAudioClassification",
        "AutoModelForConditionalGeneration",
        "AutoModel",
    ),
    Modality.CAUSAL_LM: (
        "AutoModelForCausalLM",
        "AutoModel",
    ),
    Modality.MULTIMODAL: (
        "AutoModelForImageTextToText",
        "AutoModelForMultimodalLM",
        "AutoModelForVision2Seq",
        "AutoModelForConditionalGeneration",
        "AutoModelForCausalLM",
        "AutoModel",
    ),
    Modality.SEQ2SEQ: (
        "AutoModelForSpeechSeq2Seq",
        "AutoModelForSeq2SeqLM",
        "AutoModel",
    ),
    Modality.TEXT: (
        "AutoModelForSequenceClassification",
        "AutoModelForMaskedLM",
        "AutoModel",
    ),
    Modality.VISION: (
        "AutoModelForImageClassification",
        "AutoModel",
    ),
}

_DEFAULT_CANDIDATE_AUTO_CLASSES: tuple[str, ...] = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForVision2Seq",
    "AutoModelForConditionalGeneration",
    "AutoModelForCausalLM",
    "AutoModelForSeq2SeqLM",
    "AutoModelForSequenceClassification",
    "AutoModelForPreTraining",
    "AutoModel",
)


_DEFAULT_ROPE_THETA = 10_000.0

_ROPE_THETA_ALIAS_KEYS: tuple[str, ...] = (
    "rope_theta",
    "global_rope_theta",
    "local_rope_theta",
    "rope_local_base_freq",
    "rotary_emb_base",
)


def _is_nested_rope_parameters(rope_params: Any) -> bool:
  """Returns True if `rope_params` contains per-attention-type sub-config dicts.

  Multi-attention-type configs (such as ModernBert, Gemma3, and T5Gemma2) store
  per-layer-type RoPE settings under keys like `"full_attention"` and
  `"sliding_attention"` instead of a flat `{"rope_type": ..., "rope_theta":
  ...}` mapping. Uses `any(...)` rather than `all(...)` so dicts into which
  `RotaryEmbeddingConfigMixin.standardize_rope_params()` has injected stray
  top-level `"rope_type"` or `"rope_theta"` keys are still recognized as nested.
  """
  if isinstance(rope_params, _ConfigWrapper):
    rope_params = rope_params.unwrap()
  if not isinstance(rope_params, Mapping) or not rope_params:
    return False
  return any(
      isinstance(v, (Mapping, _ConfigWrapper)) for v in rope_params.values()
  )


def _as_rope_theta(value: Any) -> float | None:
  """Converts a scalar `rope_theta` candidate to `float`, or returns `None`."""
  if isinstance(value, _ConfigWrapper):
    value = value.unwrap()
  if isinstance(value, (bool, Mapping)):
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _resolve_rope_theta(
    raw_theta: Any,
    default_theta: float | Mapping[str, float] = _DEFAULT_ROPE_THETA,
) -> float:
  """Resolves a scalar `rope_theta`, falling back to `_DEFAULT_ROPE_THETA`.

  Mapping `default_theta` values (used by hybrid-attention configs) are ignored
  here so scalar resolution falls back to `_DEFAULT_ROPE_THETA`.
  """
  for candidate in (raw_theta, default_theta):
    theta = _as_rope_theta(candidate)
    if theta is not None:
      return theta
  return _DEFAULT_ROPE_THETA


def _normalize_rope_parameters(
    params: Mapping[str, Any],
    *,
    rope_theta: float | None,
) -> dict[str, Any]:
  """Normalizes flat or nested per-attention-type `rope_parameters` dicts.

  Args:
    params: Flat RoPE mapping or nested per-attention-type RoPE mapping.
    rope_theta: Fallback scalar `rope_theta` for flat dicts when `"rope_theta"`
      is missing or `None`. Pass `None` to omit injecting a default theta and to
      drop explicit `"rope_theta": None` entries.

  Returns:
    A new normalized dictionary copy. For nested dicts, non-mapping top-level
    entries (other than `None` layer entries) are stripped and per-layer-type
    `rope_theta` values are left to `convert_rope_params_to_dict()`.
  """
  params = dict(params)
  if _is_nested_rope_parameters(params):
    # Per-layer-type theta is filled by convert_rope_params_to_dict().
    # Strip stray top-level scalar keys (such as "rope_type" or "rope_theta")
    # that standardize_rope_params() may inject when num_hidden_layers is
    # smaller than the sliding window pattern.
    normalized: dict[str, Any] = {}
    for layer_type, layer_params in params.items():
      if isinstance(layer_params, _ConfigWrapper):
        layer_params = layer_params.unwrap()
      if isinstance(layer_params, Mapping):
        sub = dict(layer_params)
        if sub.get("rope_type") is None:
          sub["rope_type"] = sub.get("type") or "default"
        if "rope_theta" in sub and sub["rope_theta"] is None:
          del sub["rope_theta"]
        normalized[layer_type] = sub
      elif layer_params is None and layer_type not in (
          "rope_theta",
          "rope_type",
      ):
        normalized[layer_type] = None
    return normalized

  if params.get("rope_type") is None:
    params["rope_type"] = params.get("type") or "default"
  if params.get("rope_theta") is None:
    if rope_theta is not None:
      params["rope_theta"] = rope_theta
    elif "rope_theta" in params:
      del params["rope_theta"]
  return params


def _has_custom_rope_converter(cfg_cls: type[Any]) -> bool:
  """Returns True if `cfg_cls` overrides `convert_rope_params_to_dict`."""
  if not _HAS_TRANSFORMERS or not hasattr(transformers, "PretrainedConfig"):
    return False
  converter = getattr(cfg_cls, "convert_rope_params_to_dict", None)
  base_converter = getattr(
      transformers.PretrainedConfig, "convert_rope_params_to_dict", None
  )
  return converter is not None and converter is not base_converter


def _sanitize_rope_parameters(
    cfg: _ConfigWrapper, *, default_theta: float = _DEFAULT_ROPE_THETA
) -> None:
  """Sanitizes rope_parameters on a wrapped config object or dict."""
  if "rope_parameters" not in cfg:
    return
  cls_default_theta = cfg.get("default_theta", default_theta)
  if isinstance(cls_default_theta, _ConfigWrapper):
    cls_default_theta = cls_default_theta.unwrap()
  fallback_theta = _resolve_rope_theta(
      cfg.get("rope_theta"),
      cls_default_theta if cls_default_theta is not None else default_theta,
  )

  rope_params = cfg.get("rope_parameters")
  raw_params = (
      rope_params.unwrap()
      if isinstance(rope_params, _ConfigWrapper)
      else rope_params
  )
  if not raw_params:
    if isinstance(cls_default_theta, Mapping):
      return
    cfg.rope_parameters = {
        "rope_type": "default",
        "rope_theta": fallback_theta,
    }
    return

  if not isinstance(raw_params, Mapping):
    return
  cfg.rope_parameters = _normalize_rope_parameters(
      raw_params, rope_theta=fallback_theta
  )


def _sanitize_common_config_dict_fields(
    config_dict: dict[str, Any],
    load_weights: bool = False,
) -> None:
  """Normalizes common schema fields on a raw config dictionary in place."""
  if _HAS_TRANSFORMERS and hasattr(
      transformers.PretrainedConfig, "_decode_special_floats"
  ):
    config_dict.update(
        transformers.PretrainedConfig._decode_special_floats(config_dict)
    )

  # Model type aliases for non-canonical community model types
  model_type = config_dict.get("model_type")
  if model_type in _MODEL_TYPE_ALIASES:
    config_dict["model_type"] = _MODEL_TYPE_ALIASES[model_type]
  elif (
      isinstance(model_type, str) and model_type.lower() in _MODEL_TYPE_ALIASES
  ):
    config_dict["model_type"] = _MODEL_TYPE_ALIASES[model_type.lower()]

  # Architecture aliases
  model_type = config_dict.get("model_type")
  if model_type in _ARCHITECTURE_ALIASES:
    aliases = _ARCHITECTURE_ALIASES[model_type]
    archs = config_dict.get("architectures")
    if isinstance(archs, list):
      config_dict["architectures"] = [aliases.get(a, a) for a in archs]

  # Defaults when field is explicitly None
  if model_type in _CONFIG_DEFAULTS_WHEN_NONE:
    for k, v in _CONFIG_DEFAULTS_WHEN_NONE[model_type].items():
      if config_dict.get(k) is None:
        config_dict[k] = v

  # Config key aliases (copy if missing)
  for src_key, dst_key in _CONFIG_KEY_ALIASES.items():
    if src_key in config_dict and dst_key not in config_dict:
      config_dict[dst_key] = config_dict[src_key]

  # Ensure return_dict is True
  if config_dict.get("return_dict") is False:
    config_dict["return_dict"] = True
  for sub in (
      "text_config",
      "speech_config",
      "decoder_config",
      "audio_config",
      "vision_config",
      "encoder",
      "decoder",
  ):
    sub_dict = config_dict.get(sub)
    if isinstance(sub_dict, dict) and sub_dict.get("return_dict") is False:
      sub_dict["return_dict"] = True

  if (
      isinstance(config_dict.get("pad_token_id"), int)
      and not isinstance(config_dict.get("pad_token_id"), bool)
      and config_dict["pad_token_id"] < 0
  ):
    config_dict["pad_token_id"] = None

  # Convert id2label keys to integer strings and values to strings
  if "id2label" in config_dict:
    id2label = config_dict["id2label"]
    if isinstance(id2label, dict):
      if all(str(k).lstrip("-").isdigit() for k in id2label.keys()):
        config_dict["id2label"] = {str(k): str(v) for k, v in id2label.items()}
      else:
        config_dict["id2label"] = {
            str(i): str(v) for i, (_, v) in enumerate(id2label.items())
        }
    elif isinstance(id2label, (list, tuple)):
      config_dict["id2label"] = {str(i): str(v) for i, v in enumerate(id2label)}

  if "label2id" in config_dict and isinstance(config_dict["label2id"], dict):
    label2id = config_dict["label2id"]
    if all(
        isinstance(v, int) or (isinstance(v, str) and v.lstrip("-").isdigit())
        for v in label2id.values()
    ):
      config_dict["label2id"] = {str(k): int(v) for k, v in label2id.items()}
    else:
      config_dict["label2id"] = {
          str(k): i for i, k in enumerate(label2id.keys())
      }

  # PreTrainedConfig.__post_init__ rejects single_label_classification with 1
  # label.
  if config_dict.get("problem_type") == "single_label_classification":
    id2label = config_dict.get("id2label")
    num_labels = (
        len(id2label)
        if isinstance(id2label, dict)
        else config_dict.get("num_labels")
    )
    if num_labels == 1:
      config_dict["problem_type"] = "regression"

  # Remove non-torch.dtype string values (e.g. "auto") before __post_init__
  for dtype_key in ("dtype", "torch_dtype"):
    dval = config_dict.get(dtype_key)
    if isinstance(dval, str):
      resolved_dtype = getattr(torch, dval, None)
      if not isinstance(resolved_dtype, torch.dtype):
        config_dict.pop(dtype_key, None)

  # Convert int dropout probabilities and scaling factors to floats
  for float_key in (
      "hidden_dropout_prob",
      "attention_probs_dropout_prob",
      "classifier_dropout",
      "routed_scaling_factor",
  ):
    val = config_dict.get(float_key)
    if isinstance(val, int) and not isinstance(val, bool):
      config_dict[float_key] = float(val)

  # wavlm, wav2vec2, etc.: num_feat_extract_layers should match len(conv_dim)
  if "conv_dim" in config_dict and isinstance(
      config_dict["conv_dim"], (list, tuple)
  ):
    config_dict["num_feat_extract_layers"] = len(config_dict["conv_dim"])

  # Strip non-mapping rope_scaling values (e.g. `"rope_scaling": false` in
  # Zyphra/ZAYA1-base) so PreTrainedConfig's `rope_scaling` property setter does
  # not overwrite `self.rope_parameters` with `False` during `__post_init__`.
  if "rope_scaling" in config_dict and not isinstance(
      config_dict["rope_scaling"], Mapping
  ):
    config_dict.pop("rope_scaling", None)

  # Normalize legacy rope_scaling dicts in place without pre-populating flat
  # rope_parameters, so multi-attention-type configs (e.g. Gemma3TextConfig,
  # Olmo3Config, ModernBertConfig) can initialize their nested
  # {"full_attention": ..., "sliding_attention": ...} structure in
  # convert_rope_params_to_dict() before applying rope_scaling.
  rope_scaling = config_dict.get("rope_scaling")
  if isinstance(rope_scaling, Mapping) and rope_scaling:
    normalized_scaling = _normalize_rope_parameters(
        rope_scaling, rope_theta=None
    )
    if _is_nested_rope_parameters(normalized_scaling) and not config_dict.get(
        "rope_parameters"
    ):
      config_dict["rope_parameters"] = normalized_scaling
      del config_dict["rope_scaling"]
    else:
      config_dict["rope_scaling"] = normalized_scaling

  # For offline execution, disable downloading pretrained backbones.
  if not load_weights and "use_pretrained_backbone" in config_dict:
    config_dict["use_pretrained_backbone"] = False

  # For offline execution, synthesize backbone_config if missing to prevent
  # network calls.
  if (
      not load_weights
      and config_dict.get("backbone")
      and not config_dict.get("backbone_config")
  ):
    if config_dict.get("use_timm_backbone", True):
      config_dict["backbone_config"] = {
          "model_type": "timm_backbone",
          "backbone": config_dict["backbone"],
          **(config_dict.get("backbone_kwargs") or {}),
      }
    else:
      backbone_name = str(config_dict["backbone"]).lower()
      config_dict["backbone_config"] = {
          "model_type": "swin" if "swin" in backbone_name else "resnet",
      }


def _sanitize_model_specific_config_dict(
    config_dict: MutableMapping[str, Any],
    load_weights: bool = False,
) -> None:
  """Applies model-type-specific normalizations to a raw config dictionary."""
  model_type = config_dict.get("model_type")
  if model_type == "xlstm":
    config_dict["chunkwise_kernel"] = "chunkwise--native_autograd"
    config_dict["sequence_kernel"] = "native_sequence__native"
    config_dict["step_kernel"] = "native"
  if not load_weights:
    if model_type == "vits":
      config_dict["use_stochastic_duration_prediction"] = False
    elif model_type == "zamba2":
      # Upstream Zamba2Model.get_layers computes `layer_id % num_mem_blocks`
      # instead of `hybrid_idx % num_mem_blocks`, leaving odd-parity hybrid
      # layers with nn.Identity() adapters when num_mem_blocks > 1.
      config_dict["num_mem_blocks"] = 1
    elif model_type in ("qwen2_5_omni", "qwen3_omni_moe"):
      config_dict["enable_audio_output"] = False

    bb_cfg = config_dict.get("backbone_config")
    if isinstance(bb_cfg, dict) and bb_cfg.get("model_type") == "timm_backbone":
      if model_type == "deformable_detr":
        num_levels = _safe_int(
            config_dict.get("num_feature_levels"), default=4, min_val=1
        )
        bb_cfg.setdefault("out_indices", [2, 3, 4] if num_levels > 1 else [4])
        if config_dict.get("dilation"):
          bb_cfg.setdefault("output_stride", 16)
      elif model_type in (
          "detr",
          "conditional_detr",
          "dab_detr",
          "table-transformer",
          "table_transformer",
          "maskformer",
      ) and not config_dict.get("backbone_kwargs"):
        bb_cfg.setdefault("out_indices", [1, 2, 3, 4])
        if config_dict.get("dilation"):
          bb_cfg.setdefault("output_stride", 16)


def _sanitize_transformers_config_dict(
    config_dict: dict[str, Any],
    load_weights: bool = False,
) -> dict[str, Any]:
  """Pre-instantiation sanitization for config dictionaries."""
  config_dict = dict(config_dict)
  _sanitize_common_config_dict_fields(config_dict, load_weights=load_weights)
  _sanitize_model_specific_config_dict(config_dict, load_weights=load_weights)
  return config_dict


def _coerce_dataclass_fields(
    cfg_cls: type[Any], cfg_dict: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
  """Coerces dataclass fields and returns (init_kwargs, extra_kwargs)."""
  known_fields = {f.name: f for f in dataclasses.fields(cfg_cls)}
  attribute_map = getattr(cfg_cls, "attribute_map", {})
  if isinstance(attribute_map, dict):
    for std_attr, model_attr in attribute_map.items():
      if (
          std_attr in known_fields
          and model_attr in cfg_dict
          and std_attr not in cfg_dict
      ):
        cfg_dict[std_attr] = cfg_dict[model_attr]

  for theta_key in _ROPE_THETA_ALIAS_KEYS:
    if theta_key in cfg_dict and cfg_dict[theta_key] is None:
      del cfg_dict[theta_key]

  if "rope_parameters" in known_fields:
    default_theta = getattr(cfg_cls, "default_theta", _DEFAULT_ROPE_THETA)
    is_hybrid_or_custom = isinstance(
        default_theta, Mapping
    ) or _has_custom_rope_converter(cfg_cls)
    rope_params = cfg_dict.get("rope_parameters")
    rope_scaling = cfg_dict.get("rope_scaling")
    if not rope_params and isinstance(rope_scaling, Mapping) and rope_scaling:
      if _is_nested_rope_parameters(rope_scaling):
        rope_params = dict(rope_scaling)
        del cfg_dict["rope_scaling"]
      elif not is_hybrid_or_custom:
        rope_params = dict(rope_scaling)

    if isinstance(default_theta, Mapping) and not _is_nested_rope_parameters(
        rope_params
    ):
      # Hybrid-attention configs build their nested rope_parameters themselves.
      if isinstance(rope_params, Mapping) and rope_params:
        logging.warning(
            "Dropping flat rope_parameters %r for hybrid-attention config %s",
            rope_params,
            cfg_cls.__name__,
        )
      cfg_dict.pop("rope_parameters", None)
      rope_params = None
    elif not rope_params:
      if is_hybrid_or_custom:
        cfg_dict.pop("rope_parameters", None)
        rope_params = None
      else:
        rope_params = {}

    if isinstance(rope_params, Mapping):
      fallback_theta = _resolve_rope_theta(
          cfg_dict.get("rope_theta"), default_theta
      )
      cfg_dict["rope_parameters"] = _normalize_rope_parameters(
          rope_params, rope_theta=fallback_theta
      )

  for fname, f in known_fields.items():
    if fname in cfg_dict:
      val = cfg_dict[fname]
      type_str = str(f.type)
      if val is None:
        if f.default is not dataclasses.MISSING:
          if "None" not in type_str:
            cfg_dict[fname] = f.default
            val = f.default
        elif f.default_factory is not dataclasses.MISSING:
          if "None" not in type_str:
            cfg_dict[fname] = f.default_factory()
            val = cfg_dict[fname]
      if isinstance(val, int) and not isinstance(val, bool):
        if "float" in type_str:
          cfg_dict[fname] = float(val)
        elif "bool" in type_str:
          cfg_dict[fname] = bool(val)

  init_kwargs = {k: v for k, v in cfg_dict.items() if k in known_fields}
  extra_kwargs = {k: v for k, v in cfg_dict.items() if k not in known_fields}
  return init_kwargs, extra_kwargs


@contextlib.contextmanager
def _suppress_strict_dataclass_validation(
    cfg_cls: type[Any],
) -> Iterator[None]:
  """Temporarily disables @strict dataclass field and class validators."""
  classes_to_patch: list[type[Any]] = list(cfg_cls.__mro__)
  if _HAS_TRANSFORMERS and hasattr(transformers, "PretrainedConfig"):
    queue = [transformers.PretrainedConfig]
    visited = set(classes_to_patch)
    while queue:
      curr = queue.pop()
      for sub in curr.__subclasses__():
        if sub not in visited:
          visited.add(sub)
          classes_to_patch.append(sub)
          queue.append(sub)

  saved_attrs: list[tuple[type[Any], str, Any]] = []
  for cls in classes_to_patch:
    if "__validators__" in cls.__dict__:
      saved_attrs.append(
          (cls, "__validators__", cls.__dict__["__validators__"])
      )
      cls.__validators__ = {}
    if "__class_validators__" in cls.__dict__:
      saved_attrs.append(
          (cls, "__class_validators__", cls.__dict__["__class_validators__"])
      )
      cls.__class_validators__ = []
  try:
    yield
  finally:
    for cls, attr_name, orig_val in saved_attrs:
      setattr(cls, attr_name, orig_val)


def _resolve_sub_config_class(
    sub_cls: Any, sub_dict: Mapping[str, Any]
) -> type[Any] | None:
  """Resolves the concrete config class for a sub-config entry."""
  if hasattr(sub_cls, "for_model") or getattr(sub_cls, "__name__", "") in (
      "AutoConfig",
      "PreTrainedConfig",
      "PretrainedConfig",
  ):
    sub_model_type = sub_dict.get("model_type")
    if isinstance(sub_model_type, str) and sub_model_type:
      return _resolve_config_class(sub_model_type, sub_dict)
    return None
  if isinstance(sub_cls, type):
    return sub_cls
  return None


def _instantiate_transformers_config_class(
    cfg_cls: type[Any],
    config_dict: dict[str, Any],
    load_weights: bool = False,
) -> Any:
  """Deterministically instantiates a Transformers config class."""
  cfg_dict = _sanitize_transformers_config_dict(
      config_dict, load_weights=load_weights
  )
  known_fields = (
      {f.name: f for f in dataclasses.fields(cfg_cls)}
      if dataclasses.is_dataclass(cfg_cls)
      else {}
  )
  sub_configs = getattr(cfg_cls, "sub_configs", {})
  if isinstance(sub_configs, dict):
    for sub_key, sub_cls in sub_configs.items():
      if sub_key in cfg_dict and isinstance(cfg_dict[sub_key], dict):
        sub_dict = _sanitize_transformers_config_dict(
            cfg_dict[sub_key], load_weights=load_weights
        )
        default_sub_kwargs = getattr(
            cfg_cls, f"_default_{sub_key}_kwargs", None
        )
        if isinstance(default_sub_kwargs, dict):
          sub_dict = {**default_sub_kwargs, **sub_dict}
        concrete_sub_cls = _resolve_sub_config_class(sub_cls, sub_dict)
        if (
            sub_key in known_fields
            and concrete_sub_cls is not None
            and getattr(cfg_cls, "__name__", "") != "Qwen3OmniMoeTalkerConfig"
        ):
          sub_inst = _instantiate_transformers_config_class(
              concrete_sub_cls, sub_dict, load_weights=load_weights
          )
          if sub_inst is not None:
            cfg_dict[sub_key] = sub_inst
        else:
          if concrete_sub_cls is not None and dataclasses.is_dataclass(
              concrete_sub_cls
          ):
            sub_init, sub_extra = _coerce_dataclass_fields(
                concrete_sub_cls, sub_dict
            )
            sub_dict = {**sub_init, **sub_extra}
          cfg_dict[sub_key] = sub_dict

  with _suppress_strict_dataclass_validation(cfg_cls):
    if dataclasses.is_dataclass(cfg_cls):
      init_kwargs, extra_kwargs = _coerce_dataclass_fields(cfg_cls, cfg_dict)
      return cfg_cls(**init_kwargs, **extra_kwargs)

    init_fn = getattr(cfg_cls, "__init__", None)
    call_kwargs = cfg_dict
    if callable(init_fn):
      sig = inspect.signature(init_fn)
      has_var_kw = any(
          p.kind == inspect.Parameter.VAR_KEYWORD
          for p in sig.parameters.values()
      )
      if not has_var_kw:
        valid_params = set(sig.parameters.keys()) - {"self", "args", "kwargs"}
        if valid_params:
          call_kwargs = {k: v for k, v in cfg_dict.items() if k in valid_params}
    return cfg_cls(**call_kwargs)


def _resolve_config_class(
    model_type: str, config_dict: Mapping[str, Any]
) -> type[Any] | None:
  """Resolves the configuration class for a model type or architecture."""
  config_mapping = getattr(
      getattr(transformers, "models", None)
      and getattr(transformers.models, "auto", None)
      and getattr(transformers.models.auto, "configuration_auto", None),
      "CONFIG_MAPPING",
      {},
  )
  cfg_cls = None
  if isinstance(config_mapping, Mapping) and model_type in config_mapping:
    cfg_cls = config_mapping[model_type]
  if cfg_cls is None:
    # Try direct config lookup (e.g. TableTransformerConfig).
    parts = model_type.replace("-", "_").split("_")
    cls_name = "".join(p.capitalize() for p in parts) + "Config"
    cfg_cls = getattr(transformers, cls_name, None)
  if isinstance(cfg_cls, str):
    cfg_cls = getattr(transformers, cfg_cls, None)
  if cfg_cls is None:
    archs = config_dict.get("architectures", [])
    if archs and isinstance(archs, list) and isinstance(archs[0], str):
      arch_cls = getattr(transformers, archs[0], None)
      if arch_cls is not None and hasattr(arch_cls, "config_class"):
        cfg_cls = getattr(arch_cls, "config_class", None)
  return cfg_cls


def _sanitize_config(config: Any) -> Any:
  """Post-instantiation, recursive over sub_configs, idempotent.

  Ensures return_dict=True, applies architecture aliases, generic rotary_dim /
  MoE top-k / conv_dim clamps, default_language, and rope_parameters defaults.
  """
  if config is None:
    return None
  cfg = _wrap_config(config)

  if cfg.get("return_dict") is False:
    cfg.return_dict = True

  model_type = cfg.get("model_type")
  if model_type in _ARCHITECTURE_ALIASES:
    aliases = _ARCHITECTURE_ALIASES[model_type]
    archs = cfg.get("architectures")
    if isinstance(archs, list):
      new_archs = [aliases.get(a, a) for a in archs]
      if new_archs != archs:
        cfg.architectures = new_archs

  if model_type in _CONFIG_DEFAULTS_WHEN_NONE:
    for k, v in _CONFIG_DEFAULTS_WHEN_NONE[model_type].items():
      if cfg.get(k) is None:
        setattr(cfg, k, v)

  # Generic rotary_dim clamp for configs with hidden_size and heads.
  rotary_dim = _safe_int(cfg.get("rotary_dim"))
  if rotary_dim:
    hidden_size = _safe_int(cfg.get("hidden_size") or cfg.get("n_embd"))
    num_heads = _safe_int(cfg.get("num_attention_heads") or cfg.get("n_head"))
    if hidden_size and num_heads and num_heads > 0:
      head_dim = hidden_size // num_heads
      if rotary_dim > head_dim:
        cfg.rotary_dim = head_dim

  # Generic MoE top-k clamp for any config with num_local_experts or num_experts
  num_local_experts = _safe_int(
      cfg.get("num_local_experts") or cfg.get("num_experts"), default=None
  )
  if num_local_experts and num_local_experts > 0:
    found_key = False
    for topk_key in _MOE_TOPK_KEYS:
      topk_val = _safe_int(cfg.get(topk_key), default=None)
      if topk_val is not None:
        found_key = True
        if topk_val > num_local_experts:
          setattr(cfg, topk_key, num_local_experts)
    if not found_key:
      topk_val = _safe_int(cfg.get("top_k"), default=None)
      if topk_val is not None and topk_val > num_local_experts:
        cfg.top_k = num_local_experts

  # Generic conv_dim clamp
  conv_dim = cfg.get("conv_dim")
  if isinstance(conv_dim, (list, tuple)):
    cfg.num_feat_extract_layers = len(conv_dim)

  # Generic default_language for configs with languages list.
  langs = cfg.get("languages")
  if (
      langs
      and isinstance(langs, (list, tuple))
      and not cfg.get("default_language")
  ):
    cfg.default_language = langs[0]

  _sanitize_rope_parameters(cfg)

  pad_id = cfg.get("pad_token_id")
  if isinstance(pad_id, int) and not isinstance(pad_id, bool) and pad_id < 0:
    cfg.pad_token_id = None

  text_cfg = cfg.get("text_config")
  if text_cfg is not None:
    text_vocab_size = _safe_int(
        text_cfg.get("vocab_size"), default=None, min_val=2
    )
    if text_vocab_size is not None:
      for tok_key in ("image_token_id", "image_token_index"):
        tok_val = _safe_int(cfg.get(tok_key), default=None, min_val=0)
        if tok_val is not None and tok_val >= text_vocab_size:
          setattr(cfg, tok_key, text_vocab_size - 1)

  # Recursively sanitize sub-configs
  sub_configs = cfg.get("sub_configs")
  if isinstance(sub_configs, dict):
    for sub_key in sub_configs:
      sub = cfg.get(sub_key)
      if sub is not None:
        _sanitize_config(sub)

  for sub_name in (
      "text_config",
      "speech_config",
      "decoder_config",
      "audio_config",
      "vision_config",
  ):
    if not isinstance(sub_configs, dict) or sub_name not in sub_configs:
      sub = cfg.get(sub_name)
      if sub is not None:
        _sanitize_config(sub)

  return config


def _load_and_sanitize_transformers_config(
    config_path: epath.PathLike | str,
    load_weights: bool = False,
) -> Any:
  """Stage 2: Deterministically loads, sanitizes, and constructs a model config.

  What this stage does:
    Reads the raw `config.json` from `config_path`, normalizes common and
    model-specific dictionary fields (`_sanitize_transformers_config_dict`),
    resolves and instantiates the strongly-typed `PretrainedConfig` subclass
    (`_resolve_config_class` and `_instantiate_transformers_config_class`) with
    `@strict` dataclass validation suppressed, and applies recursive
    post-instantiation invariant normalization (`_sanitize_config`).

  Why this is needed as a separate stage:
    Upstream `AutoConfig.from_pretrained` couples file resolution, JSON loading,
    and strict `@strict` dataclass validation into a single monolithic call.
    Because ~60+ cached `config.json` files in the shared weights cache contain
    legacy or community schema discrepancies (such as `null` attributes where
    `int`/`bool` is required by `@strict`, `int` values for `float` fields, or
    online TIMM backbone triggers), calling `AutoConfig.from_pretrained`
    directly caused crashes that previously required a multi-tier `try / except`
    fallback chain. Executing Stage 2 as a dedicated deterministic
    transformation pipeline guarantees that every raw `config.json` is
    normalized before and after `PretrainedConfig` construction without any
    trial-and-error exception catching.

  Example:
    `deepseek-ai/DeepSeek-V2-Lite` stores `"routed_scaling_factor": 1` (an
    `int`) in its `config.json`, which causes HuggingFace v5's `@strict`
    dataclass validator on `DeepseekV2Config` to raise
    `StrictDataclassFieldValidationError` because the field is annotated as
    `float`. Similarly, `facebook/detr-resnet-50` sets
    `"use_pretrained_backbone": true` and `"backbone": "resnet50"`, which
    triggers an online TIMM hub download during `DetrConfig.__init__` when
    `load_weights=False`. Stage 2 deterministically converts
    `routed_scaling_factor` to `1.0` and synthesizes an offline
    `TimmBackboneConfig` with `use_pretrained_backbone=False` before
    instantiating the config class.

  Args:
    config_path: Path to `config.json` or a directory containing `config.json`.
    load_weights: Whether pretrained weights will be loaded.

  Returns:
    A sanitized `_ConfigWrapper` wrapping the instantiated `PretrainedConfig`
    (or config dictionary for unregistered custom model types).
  """
  path = epath.Path(config_path)
  cfg_file = (
      path / "config.json" if _RETRY_TRANSIENT_IO(path.is_dir)() else path
  )
  raw_dict = json.loads(_RETRY_TRANSIENT_IO(cfg_file.read_text)())

  config_dict = _sanitize_transformers_config_dict(
      raw_dict, load_weights=load_weights
  )
  model_type = config_dict.get("model_type")
  if not isinstance(model_type, str) or not model_type:
    raise ValueError(
        f"Missing or invalid model_type in config for {config_path}"
    )

  cfg_cls = _resolve_config_class(model_type, config_dict)
  if cfg_cls is not None:
    inst = _instantiate_transformers_config_class(
        cfg_cls, config_dict, load_weights=load_weights
    )
    if inst is not None:
      return _sanitize_config(_wrap_config(inst))

  return _sanitize_config(_wrap_config(config_dict))


def _resolve_architecture_class(
    arch_name: str | None, model_type: str = ""
) -> type[Any] | None:
  """Resolves the architecture model class from name and model type."""
  if not arch_name or not _HAS_TRANSFORMERS:
    return None

  # 1. Top-level exact match
  if hasattr(transformers, arch_name):
    return getattr(transformers, arch_name)

  # 2. Top-level case-insensitive match
  arch_lower = arch_name.lower()
  for attr in dir(transformers):
    if attr.lower() == arch_lower:
      return getattr(transformers, attr)

  # 3. Model submodule lookup via transformers.models lazy module
  models_pkg = getattr(transformers, "models", None)
  if models_pkg is not None and model_type:
    candidate_types = [model_type.replace("-", "_")]
    for sep in ("_", "-"):
      if sep in model_type:
        base_type = model_type.rsplit(sep, 1)[0].replace("-", "_")
        if base_type not in candidate_types:
          candidate_types.append(base_type)
    for cand_type in candidate_types:
      mod = getattr(models_pkg, cand_type, None)
      if mod is not None:
        if hasattr(mod, arch_name):
          return getattr(mod, arch_name)
        modeling_sub = getattr(mod, f"modeling_{cand_type}", None)
        if modeling_sub is not None and hasattr(modeling_sub, arch_name):
          return getattr(modeling_sub, arch_name)

  return None


def _is_arch_class_compatible_with_config(
    arch_cls: type[Any], raw_cfg: Any
) -> bool:
  """Returns True if arch_cls can be instantiated directly from raw_cfg."""
  expected_cfg_cls = getattr(arch_cls, "config_class", None)
  if not isinstance(expected_cfg_cls, type) or not isinstance(
      raw_cfg, getattr(transformers, "PretrainedConfig", ())
  ):
    return True
  if isinstance(raw_cfg, expected_cfg_cls):
    return True
  expected_sub_configs = getattr(expected_cfg_cls, "sub_configs", None)
  if isinstance(expected_sub_configs, dict) and expected_sub_configs:
    # A composite model class (e.g. Gemma3ForConditionalGeneration) requires
    # sub-configs (e.g. vision_config) that a standalone text config lacks.
    if any(getattr(raw_cfg, k, None) is None for k in expected_sub_configs):
      return False
  return True


def _resolve_transformers_model_class(
    config: Any, modality: Modality
) -> type[torch.nn.Module]:
  """Stage 3: Deterministically resolves the target `nn.Module` model class.

  What this stage does:
    Resolves the concrete HuggingFace `PreTrainedModel` class from the sanitized
    `config` and `modality` by (1) checking `config.architectures[0]` via
    `_resolve_architecture_class` and verifying config compatibility, or (2)
    looking up `type(config)` in the `_model_mapping` registry of the modality's
    prioritized `AutoModel*` classes (`_CANDIDATE_AUTO_CLASSES_BY_MODALITY`).

  Why this is needed as a separate stage:
    Separating class resolution from model instantiation allows the pipeline to
    determine the exact `nn.Module` class without speculatively constructing
    multiple `AutoModel*` instances inside a `try / except` loop. Furthermore,
    checking `_is_arch_class_compatible_with_config` catches mismatches between
    composite architecture names and standalone sub-configs before
    `model_cls._from_config` is ever invoked.

  Example:
    `FacebookAI/roberta-large-mnli` specifies
    `architectures: ["RobertaForSequenceClassification"]`, which Stage 3
    resolves directly from `transformers`. By contrast,
    `TrueHealth/MedImageInsight` has no `architectures` field in its
    `config.json` and uses `model_type: "uni_cl"`; Stage 3 inspects
    `_model_mapping` across `Modality.VISION`'s candidate `AutoModel` registries
    to deterministically select the registered vision model class without trial
    instantiations. And when a standalone `Gemma3TextConfig` still lists
    `architectures: ["Gemma3ForConditionalGeneration"]` (whose `config_class` is
    the composite `Gemma3Config` requiring a `vision_config`), Stage 3 detects
    the config class mismatch and resolves `Gemma3ForCausalLM` via
    `AutoModelForCausalLM._model_mapping[Gemma3TextConfig]`.

  Args:
    config: Sanitized `_ConfigWrapper` or `PretrainedConfig`.
    modality: The detected `Modality` of the model.

  Returns:
    The resolved `torch.nn.Module` subclass.

  Raises:
    ValueError: If no registered model class matches `config` and `modality`.
  """
  cfg = _wrap_config(config)
  raw_cfg = cfg.unwrap() if isinstance(cfg, _ConfigWrapper) else cfg
  architectures = cfg.get("architectures", [])
  arch_name = architectures[0] if architectures else None
  model_type = cfg.get("model_type", "")

  arch_cls = _resolve_architecture_class(arch_name, model_type)
  if arch_cls is not None and _is_arch_class_compatible_with_config(
      arch_cls, raw_cfg
  ):
    return arch_cls

  candidate_names = _CANDIDATE_AUTO_CLASSES_BY_MODALITY.get(
      modality, _DEFAULT_CANDIDATE_AUTO_CLASSES
  )
  for class_name in tuple(candidate_names) + ("AutoModel",):
    if class_name == "AutoModelForCTC" and not cfg.get("vocab_size"):
      continue
    auto_cls = getattr(transformers, class_name, None)
    mapping = getattr(auto_cls, "_model_mapping", None)
    if mapping is not None and type(raw_cfg) in mapping:
      resolved = mapping[type(raw_cfg)]
      if isinstance(resolved, (list, tuple)):
        return resolved[0]
      return resolved

  if arch_cls is not None:
    return arch_cls

  raise ValueError(
      f"Cannot resolve Transformers model class for model_type={model_type!r},"
      f" architectures={architectures!r}, modality={modality}."
  )


def _delegate_missing_forward(model: torch.nn.Module) -> None:
  """Binds forward to first child that accepts input_ids if needed."""
  if model is None or type(model).forward is not torch.nn.Module.forward:
    return
  for child in model.children():
    forward_fn = getattr(child, "forward", None)
    if forward_fn is not None:
      sig = inspect.signature(forward_fn)
      if "input_ids" in sig.parameters:

        def _delegated_forward(self, *args, **kwargs):
          return child(*args, **kwargs)

        model.forward = types.MethodType(_delegated_forward, model)
        return


def _apply_model_patches(
    model: torch.nn.Module | None, config: Any
) -> torch.nn.Module | None:
  """Applies post-instantiation patches to the model."""
  if model is None:
    return None
  cfg = _wrap_config(config)

  if cfg.get("model_type", "") in _ZERO_INIT_WEIGHT_MODEL_TYPES:

    def _zero_weights_pre_hook(module, *_unused_args, **_unused_kwargs):
      for p in module.parameters():
        if p.device.type != "meta" and p.dtype.is_floating_point:
          p.data.zero_()

    # pyrefly: ignore[missing-attribute]
    model.register_forward_pre_hook(_zero_weights_pre_hook)

  if (
      cfg.get("model_type", "") == "bloom"
      or "bloom" in str(type(model)).lower()
  ):
    for sub in model.modules():
      if sub.__class__.__name__ == "BloomAttention":
        cls = sub.__class__
        if not getattr(cls, "_alibi_patched", False):
          orig_forward = cls.forward

          def _safe_bloom_forward(self, *args, **kwargs):
            alibi = kwargs.get("alibi")
            layer_past = kwargs.get("layer_past")
            alibi_arg_idx = None
            if alibi is None and len(args) > 2:
              alibi = args[2]
              alibi_arg_idx = 2
            if layer_past is None and len(args) > 4:
              layer_past = args[4]
            if alibi is not None and layer_past is not None:
              max_len = getattr(layer_past, "max_cache_len", None)
              if max_len is not None and alibi.shape[-1] < max_len:
                padded_alibi = torch.nn.functional.pad(
                    alibi, (0, max_len - alibi.shape[-1])
                )
                if "alibi" in kwargs:
                  kwargs["alibi"] = padded_alibi
                elif alibi_arg_idx is not None:
                  args_list = list(args)
                  args_list[alibi_arg_idx] = padded_alibi
                  args = tuple(args_list)
            return orig_forward(self, *args, **kwargs)

          cls.forward = _safe_bloom_forward
          cls._alibi_patched = True
        break

  if hasattr(model, "set_default_language") and hasattr(model, "config"):
    langs = getattr(model.config, "languages", None)
    if (
        langs
        and isinstance(langs, (list, tuple))
        and isinstance(langs[0], str)
        and not getattr(model.config, "default_language", None)
    ):
      set_default_lang = getattr(model, "set_default_language")
      if callable(set_default_lang):
        # pyrefly: ignore[missing-attribute]
        set_default_lang(langs[0])

  if (
      model.__class__.__name__ == "BarkModel"
      or cfg.get("model_type", "") == "bark"
  ):

    def _bark_forward(self, *args, **kwargs):
      if hasattr(self, "semantic"):
        return self.semantic(*args, **kwargs)
      return None

    model.forward = types.MethodType(_bark_forward, model)

  if (
      "Qwen2_5Omni" in model.__class__.__name__
      or "Qwen3Omni" in model.__class__.__name__
      or "qwen2_5_omni" in cfg.get("model_type", "").lower()
      or "qwen3_omni" in cfg.get("model_type", "").lower()
  ):
    if hasattr(model, "thinker"):

      def _qwen_omni_forward(self, *args, **kwargs):
        return self.thinker(*args, **kwargs)

      model.forward = types.MethodType(_qwen_omni_forward, model)

  if (
      "Sam2Video" in model.__class__.__name__
      or "EdgeTamVideo" in model.__class__.__name__
      or cfg.get("model_type", "").lower() in ("sam2_video", "edgetam_video")
  ):
    if hasattr(model, "get_image_features"):

      def _sam2_video_forward(self, pixel_values=None, *args, **kwargs):
        return self.get_image_features(
            pixel_values=pixel_values, *args, **kwargs
        )

      model.forward = types.MethodType(_sam2_video_forward, model)

  if (
      "Sam3Video" in model.__class__.__name__
      or cfg.get("model_type", "").lower() == "sam3_video"
  ):
    detector = getattr(model, "detector_model", None)
    if detector is not None and hasattr(detector, "get_vision_features"):

      def _sam3_video_forward(self, pixel_values=None, *args, **kwargs):
        if pixel_values is not None:
          vision_embeds = self.detector_model.get_vision_features(
              pixel_values=pixel_values
          )
          if hasattr(self, "get_vision_features_for_tracker"):
            self.get_vision_features_for_tracker(vision_embeds)
          return vision_embeds
        return self.detector_model(*args, **kwargs)

      model.forward = types.MethodType(_sam3_video_forward, model)

  if (
      "DeformableDetr" in model.__class__.__name__
      or "deformable_detr" in cfg.get("model_type", "").lower()
      or "OneFormer" in model.__class__.__name__
      or "oneformer" in cfg.get("model_type", "").lower()
  ):
    orig_fwd = model.__class__.forward

    def _safe_deformable_detr_forward(self, *args, **kwargs):
      pixel_values = kwargs.get("pixel_values")
      if pixel_values is None and args:
        pixel_values = args[0]
      if pixel_values is not None and getattr(pixel_values, "is_meta", False):
        self.to(pixel_values.device)
      return orig_fwd(self, *args, **kwargs)

    model.forward = types.MethodType(_safe_deformable_detr_forward, model)

  _delegate_missing_forward(model)
  return model


def _instantiate_transformers_model(
    model_cls: type[torch.nn.Module],
    config: Any,
    dtype: torch.dtype | None,
) -> torch.nn.Module:
  """Stage 4: Deterministically instantiates and patches the resolved model.

  What this stage does:
    Unwraps the sanitized `config` (and extracts `config.get_text_config()` when
    `model_cls` expects the inner text sub-config of a composite multimodal
    config), constructs the model in `dtype` via
    `model_cls._from_config(target_cfg, dtype=dtype)`, casts any remaining
    floating-point parameters/buffers to it, and applies post-instantiation
    runtime hooks via `_apply_model_patches`.

  Why this is needed as a separate stage:
    Model construction must be deferred inside `ModuleSpec.module_factory` so
    that callers (such as `meta_test` running inside `with torch.device("meta")`
    or `cpu_test` on CPU) control the active PyTorch device context at
    instantiation time. Isolating construction and patching in Stage 4 also
    handles runtime invariants that cannot be expressed in `PretrainedConfig`
    alone, such as composite-to-text config unwrapping, zeroing uninitialized
    parameters for numerically sensitive models, or patching attention forward
    methods.

  Example:
    `mistralai/Ministral-3-8B-Instruct-2512` has a composite `Mistral3Config`
    (containing both `text_config` and `vision_config`), while its resolved
    causal LM class expects the inner `Ministral3Config`
    (`config.get_text_config()`); Stage 4 unwraps
    `target_cfg = raw_cfg.get_text_config()` before calling `_from_config`. As
    another example, `kakao-enterprise/vits-ljs` (`model_type="vits"`) and
    `google/tapas-base` (`model_type="tapas"`) produce `NaN` or `IndexError`
    during a dry-run forward pass when initialized with random uncalibrated
    weights on CPU; Stage 4 registers a forward pre-hook via
    `_apply_model_patches` to zero-initialize their floating-point parameters
    prior to `forward()`.

  Args:
    model_cls: The resolved `torch.nn.Module` class from Stage 3.
    config: The sanitized `_ConfigWrapper` or `PretrainedConfig` from Stage 2.
    dtype: The dtype to construct the model in. If None, the model is
      constructed in the config dtype, or in the default dtype if the config has
      none.

  Returns:
    The instantiated and patched `torch.nn.Module`.

  Raises:
    ValueError: If `dtype` is not a floating point dtype.
  """
  if dtype is not None and not dtype.is_floating_point:
    raise ValueError(f"dtype must be a floating point dtype, got {dtype}.")
  raw_cfg = config.unwrap() if isinstance(config, _ConfigWrapper) else config
  target_cfg = raw_cfg

  expected_cfg_cls = getattr(model_cls, "config_class", None)
  sub_configs = getattr(raw_cfg, "sub_configs", None)
  if (
      isinstance(expected_cfg_cls, type)
      and isinstance(sub_configs, dict)
      and not isinstance(raw_cfg, expected_cfg_cls)
      and sub_configs.get("text_config") == expected_cfg_cls
      and hasattr(raw_cfg, "get_text_config")
  ):
    target_cfg = raw_cfg.get_text_config()
  # `_from_config` writes the dtype to the config and its sub-configs. Keep
  # the config of the module spec, which later calls and the input factory
  # read, unchanged.
  target_cfg = copy.deepcopy(target_cfg)

  if callable(getattr(model_cls, "_from_config", None)):
    # Build the model directly in `dtype`, which `_from_config` also sets as
    # the default dtype during construction. Without an explicit dtype,
    # `_from_config` uses the root config's dtype for the whole model. Some
    # configs set a dtype only on a sub-config (e.g. Qwen3.5-397B-A17B sets it
    # only on `text_config`), so the root dtype is None and the model is built
    # in float32 and then cast, which doubles peak memory.
    from_config_kwargs = {} if dtype is None else {"dtype": dtype}
    # pyrefly: ignore[missing-attribute]
    model = model_cls._from_config(target_cfg, **from_config_kwargs)
  else:
    # Classes without `_from_config` (e.g. `AutoModel*` classes named in
    # `architectures`) take no dtype, so construct them under it.
    with _default_dtype(dtype):
      if callable(getattr(model_cls, "from_config", None)):
        # pyrefly: ignore[missing-attribute]
        model = model_cls.from_config(target_cfg)
      else:
        model = model_cls(target_cfg)

  if model is not None and dtype is not None:
    model = model.to(dtype=dtype)  # pyrefly: ignore[missing-attribute]
  patched = _apply_model_patches(model, config)
  assert patched is not None
  return patched


class TransformersProvider(BaseProvider):
  """Provider for Hugging Face Transformers models.

  Note:
    This provider relies on local resource files for model configurations
    rather than downloading directly from the Hugging Face Hub.
  """

  _FILES = (
      resources.files("perflab").joinpath(
          "huggingface_transformers/model_configs"
      ),
  )

  def __init__(
      self, base_path: str | None = None, subdir: str = "transformers"
  ):
    super().__init__(base_path=base_path, subdir=subdir)

  @classmethod
  def in_tree_config(cls, name: str) -> resources_abc.Traversable | None:
    """Returns the in-tree `config.json` of a model, or None if there is none.

    These configs are the bundled package resources (`_FILES`) that the provider
    reads when the model is not in the shared weights cache, before it falls
    back to GCS.

    Args:
      name: Model identifier formatted as `{owner}/{model}`.
    """
    rel_cfg = str(pathlib.Path(name) / "config.json")
    for files_root in cls._FILES:
      traversable = files_root.joinpath(rel_cfg)
      if traversable.is_file():
        return traversable
    return None

  def list_modules(self) -> Sequence[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/transformers/{owner}/{model}/files

    Returns:
      A list of model name strings formatted as '{owner}/{model}'.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    modules = []
    for p in self._base_path.glob("*/*/"):
      modules.append(p.relative_to(self._base_path).as_posix())
    return modules

  @contextlib.contextmanager
  def _resolve_transformers_config_path(
      self, name: str, load_weights: bool = False
  ) -> Iterator[tuple[epath.Path, epath.Path | None]]:
    """Stage 1: Deterministically resolves the config path across storage tiers.

    What this stage does:
      Locates `config.json` and the optional model directory across three
      storage tiers: bundled package resources (`_FILES`, checked first when
      `load_weights=False` to avoid remote filesystem I/O), the shared weights
      cache (`{base_path}/transformers/{name}/config.json`), or a GCS bucket
      download (`fetch_gcs_file`).

    Why this is needed as a separate stage:
      Separating file location discovery from JSON parsing and config
      instantiation isolates filesystem I/O and resource context management
      (`resources.as_file`) from schema validation. If a config file is missing
      or `load_weights=True` is requested without a local cache directory, this
      stage fails immediately with a clear `ValueError` instead of conflating
      missing files with malformed config schemas.

    Example:
      When loading `google/gemma-2-2b` in unit tests with `load_weights=False`,
      Stage 1 locates `google/gemma-2-2b/config.json` inside the bundled
      `google/examples/huggingface_transformers/model_configs` package resource
      without probing remote storage, and yields the extracted path alongside
      `default_model_dir`. Conversely, when `load_weights=True` with
      `/path/to/weights` mounted, Stage 1 yields
      `/path/to/weights/transformers/google/gemma-2-2b/config.json` and the
      cache directory as `model_dir_or_repo_id` so Stage 4 and Stage 5 can load
      weights and tokenizers from the same directory.

    Args:
      name: Model identifier formatted as `{owner}/{model}`.
      load_weights: Whether pretrained weights are required.

    Yields:
      A tuple `(config_path, model_dir_or_repo_id)`.

    Raises:
      ValueError: If `load_weights=True` without a cached model directory, or if
        `config.json` is not found in any storage tier.
    """
    default_model_dir: epath.Path | None = None
    if self.has_cache_dir and self._base_path is not None:
      default_model_dir = self._base_path / name

    if not load_weights:
      traversable = self.in_tree_config(name)
      if traversable is not None:
        with resources.as_file(traversable) as local_file:
          if local_file.exists():
            yield epath.Path(local_file), default_model_dir
            return

    if default_model_dir is not None:
      cfg_file = default_model_dir / "config.json"
      try:
        if _RETRY_TRANSIENT_IO(cfg_file.exists)():
          yield cfg_file, default_model_dir
          return
      except (OSError, gcp_exceptions.RetryError) as exc:
        if load_weights:
          raise
        logging.warning(
            "Falling back to GCS for %s after cache check error: %s", name, exc
        )

    if load_weights:
      raise ValueError(
          f"load_weights cannot be set to True for {name} when falling back"
          " to local configuration resources or GCS."
      )

    dest_config = self.fetch_gcs_file(f"{name}/config.json")
    if dest_config is not None and dest_config.exists():
      yield epath.Path(dest_config), epath.Path(dest_config.parent)
      return

    rel_cfg = str(pathlib.Path(name) / "config.json")
    files_str = ", ".join(str(r.joinpath(rel_cfg)) for r in self._FILES)
    raise ValueError(
        f"Model config for '{name}' is missing in local resources"
        f" ({files_str})"
        " and could not be loaded from cache or GCS."
    )

  def _load_config_and_model_dir(
      self, name: str, load_weights: bool = False
  ) -> tuple[Any, epath.Path | None]:
    base_path = self._base_path if self.has_cache_dir else None

    def _loader() -> module_cache.ConfigCacheEntry:
      with self._resolve_transformers_config_path(
          name, load_weights=load_weights
      ) as (cfg_path, model_dir_or_repo_id):
        config = _load_and_sanitize_transformers_config(
            cfg_path, load_weights=load_weights
        )
        return module_cache.ConfigCacheEntry(
            config=config,
            path=model_dir_or_repo_id,
        )

    entry = module_cache.CONFIG_CACHE.get(
        "transformers",
        base_path,
        name,
        load_weights=load_weights,
        loader=_loader,
    )
    return entry.config, entry.path

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a Transformer model.

    Executes the 5-stage deterministic pipeline:
      Stage 1: `_resolve_transformers_config_path`
      Stage 2: `_load_and_sanitize_transformers_config`
      Stage 3: `_resolve_transformers_model_class`
      Stage 4: `_instantiate_transformers_model`
      Stage 5: `_load_transformers_preprocessor`

    Args:
      name: Name (str) of the hf transformer model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable that accepts and returns a
        transformers.PretrainedConfig object to modify the model configuration.
      **kwargs: Additional keyword arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    # Stages 1 & 2: Resolve config path and load/sanitize config.
    config, model_dir_or_repo_id = self._load_config_and_model_dir(
        name, load_weights=load_weights
    )

    if modify_config_hook is not None:
      if load_weights:
        raise NotImplementedError(
            "modify_config_hook is not supported when load_weights is True."
        )
      config = _sanitize_config(_wrap_config(modify_config_hook(config)))

    # When loading models offline (load_weights=False), some vision models (like
    # DETR) will still try to download pretrained backbones (e.g. via TIMM)
    # over the network. Disabling this forces offline initialization with
    # random weights, avoiding network errors on Forge.
    if not load_weights and "use_pretrained_backbone" in config:
      config.use_pretrained_backbone = False

    if not load_weights and config.get("model_type", "") == "vits":
      config.use_stochastic_duration_prediction = False

    # TAPAS's relative position embeddings per table cell call `Tensor.item()`,
    # which fails on the meta device. Set these on the config here rather than
    # in the input factory, since models are constructed from a copy of it.
    if not load_weights and config.get("model_type", "") == "tapas":
      config.reset_position_index_per_cell = False
      config.init_cell_selection_weights_to_zero = False

    modality = _determine_modality(config)
    # `_extract_target_dtype` returns float32 also for configs without a dtype,
    # which Transformers constructs in the default dtype. Some callers set it to
    # their run dtype, so don't override it with float32.
    target_dtype = _extract_target_dtype(config)
    config_dtype = None if target_dtype == torch.float32 else target_dtype

    if load_weights:
      if modality == Modality.CAUSAL_LM:
        model_cls = transformers.AutoModelForCausalLM
      elif modality == Modality.SEQ2SEQ:
        if "whisper" in config.get("model_type", ""):
          model_cls = getattr(
              transformers, "AutoModelForSpeechSeq2Seq", transformers.AutoModel
          )
        else:
          model_cls = transformers.AutoModelForSeq2SeqLM
      else:
        model_cls = transformers.AutoModel

      kwargs.setdefault("low_cpu_mem_usage", True)
      kwargs.setdefault("ignore_mismatched_sizes", True)
      # `torch_dtype` is deprecated in favor of `dtype`, which takes precedence
      # in `from_pretrained`. Keep a caller's `torch_dtype` over `config_dtype`.
      if "torch_dtype" in kwargs:
        kwargs.setdefault("dtype", kwargs.pop("torch_dtype"))
      if config_dtype is not None:
        kwargs.setdefault("dtype", config_dtype)
      # Pyrefly cannot narrow `model_cls`, which is assigned in branches above,
      # inside the closure.
      from_pretrained = model_cls.from_pretrained

      def model_fn(dtype: torch.dtype | None = None) -> torch.nn.Module:
        pretrained_kwargs = (
            kwargs if dtype is None else {**kwargs, "dtype": dtype}
        )
        return from_pretrained(str(model_dir_or_repo_id), **pretrained_kwargs)

    else:
      # Stage 3 & Stage 4: Resolve model class and instantiate model.
      resolved_cls = _resolve_transformers_model_class(config, modality)

      def model_fn(dtype: torch.dtype | None = None) -> torch.nn.Module:
        return _instantiate_transformers_model(
            resolved_cls, config, config_dtype if dtype is None else dtype
        )

    # Stage 5: Preprocessor factory.
    preprocessor_fn = lambda: _load_transformers_preprocessor(
        model_dir_or_repo_id, modality=modality
    )

    def _input_fn(
        shape=None, device="cpu"
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      model_dir = (
          str(model_dir_or_repo_id)
          if (
              modality == Modality.VISION  # Only VISION uses model_dir.
              and model_dir_or_repo_id is not None
              and _RETRY_TRANSIENT_IO(model_dir_or_repo_id.exists)()
          )
          else None
      )
      input_kwargs = transformers_inputs.generate_transformers_inputs(
          config,
          modality,
          shape,
          device,
          model_dir,
          preprocessor_loader=lambda d: _load_transformers_preprocessor(
              d, modality=Modality.VISION
          ),
      )
      return (), input_kwargs

    return ModuleSpec(
        model_fn, _input_fn, preprocessor_fn, config, modality=modality
    )


_DIFFUSERS_SUBFOLDER_PRIORITY: tuple[str, ...] = (
    "transformer",
    "unet",
    "prior",
    "controlnet",
    "movq",
    "vae",
    "vqvae",
)

_DEFAULT_DIFFUSERS_SUBFOLDERS: tuple[str | None, ...] = (
    "transformer",
    "unet",
    None,
    "vae",
    "vqvae",
    "movq",
)


def _extract_diffusers_subfolder_candidates(
    index_dict: Mapping[str, Any],
) -> Sequence[str]:
  """Extracts prioritized candidate subfolders from diffusers model_index."""
  candidates = []
  for key in _DIFFUSERS_SUBFOLDER_PRIORITY:
    val = index_dict.get(key)
    if val and isinstance(val, (list, tuple)) and val[0] is not None:
      candidates.append(key)
      break
  return candidates


def _resolve_diffusers_model_class(
    name: str, config_dict: Mapping[str, Any]
) -> tuple[str, type[Any]]:
  """Deterministically resolves the Diffusers model class name and class."""
  class_name = config_dict.get("_class_name")
  if not class_name:
    if "transformer" in name.lower() or "cosmos" in name.lower():
      class_name = "Transformer2DModel"
    else:
      class_name = "UNet2DConditionModel"
  model_cls = getattr(diffusers, class_name, None)
  if model_cls is None:
    logging.warning(
        "Could not find diffusers model class %s, falling back to"
        " UNet2DConditionModel.",
        class_name,
    )
    model_cls = diffusers.UNet2DConditionModel
  if hasattr(diffusers, "models") and hasattr(
      diffusers.models, "model_loading_utils"
  ):
    remapping_dict = getattr(
        diffusers.models.model_loading_utils, "_CLASS_REMAPPING_DICT", {}
    )
    if class_name in remapping_dict:
      norm_type = config_dict.get("norm_type")
      remapped = remapping_dict[class_name].get(norm_type)
      if remapped and hasattr(diffusers, remapped):
        class_name = remapped
        model_cls = getattr(diffusers, remapped)
  return class_name, model_cls


class DiffusersProvider(BaseProvider):
  """Provider for Hugging Face Diffusers models.

  Note:
    This provider relies on local resource files for model configurations
    rather than downloading directly from the Hugging Face Hub.
  """

  _FILES = resources.files("perflab").joinpath(
      "huggingface_diffusers/model_configs"
  )

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="diffusers")

  def list_modules(self) -> Sequence[str]:
    """Lists the names of all models available from this provider.

    Structured as {base_path}/diffusers/{owner}/{model}/files

    Returns:
      A list of model name strings formatted as '{owner}/{model}'.
    """
    if not self.has_cache_dir:
      return []
    assert self._base_path is not None
    modules = []
    for p in self._base_path.glob("*/*/"):
      modules.append(p.relative_to(self._base_path).as_posix())
    return modules

  def _load_config(
      self,
      name: str,
      subfolder: str | None = None,
      load_weights: bool = False,
  ) -> tuple[Mapping[str, Any], str | None, epath.Path | None]:
    base_path = self._base_path if self.has_cache_dir else None

    def _loader() -> module_cache.ConfigCacheEntry:
      raw_config = None
      resolved_subfolder = subfolder
      model_path = None

      if subfolder:
        candidate_subfolders: list[str | None] = [subfolder]
      else:
        candidate_subfolders = []
        if self.has_cache_dir and self._base_path is not None:
          model_path = self._base_path / name
          if model_path.exists() and model_path.is_dir():
            index_path = model_path / "model_index.json"
            if index_path.exists():
              with index_path.open("r", encoding="utf-8") as fp:
                candidate_subfolders.extend(
                    _extract_diffusers_subfolder_candidates(json.load(fp))
                )
            for sub_dir in model_path.iterdir():
              if sub_dir.is_dir():
                for sub_target in ("transformer", "unet"):
                  if (sub_dir / sub_target / "config.json").exists():
                    nested_cand = f"{sub_dir.name}/{sub_target}"
                    if nested_cand not in candidate_subfolders:
                      candidate_subfolders.append(nested_cand)
        for cand in _DEFAULT_DIFFUSERS_SUBFOLDERS:
          if cand not in candidate_subfolders:
            candidate_subfolders.append(cand)

      if self.has_cache_dir and self._base_path is not None:
        model_path = self._base_path / name
        if model_path.exists() and model_path.is_dir():
          for cand in candidate_subfolders:
            cfg_file = (
                model_path / cand / "config.json"
                if cand
                else model_path / "config.json"
            )
            if cfg_file.exists():
              raw_config = auto_model.AutoModel.load_config(
                  str(model_path), subfolder=cand
              )
              resolved_subfolder = cand
              break

      if raw_config is None and not load_weights:
        model_dir = pathlib.Path(name)
        local_candidates = list(candidate_subfolders)
        index_file = self._FILES.joinpath(str(model_dir / "model_index.json"))
        if index_file.is_file():
          with resources.as_file(index_file) as f:
            if f.is_file():
              with open(f, "r", encoding="utf-8") as fp:
                for key in _extract_diffusers_subfolder_candidates(
                    json.load(fp)
                ):
                  if key not in local_candidates:
                    local_candidates.insert(0, key)

        for cand in local_candidates:
          target_rel = (
              model_dir / cand / "config.json"
              if cand
              else model_dir / "config.json"
          )
          traversable = self._FILES.joinpath(str(target_rel))
          if traversable.is_file():
            with resources.as_file(traversable) as f:
              if f.is_file():
                raw_config = auto_model.AutoModel.load_config(str(f.parent))
                resolved_subfolder = cand
                break

      if raw_config is None and not load_weights:
        gcs_candidates: list[str | None] = []
        dest_index = self.fetch_gcs_file(f"{name}/model_index.json")
        if dest_index and dest_index.exists():
          with open(dest_index, "r", encoding="utf-8") as fp:
            gcs_candidates.extend(
                _extract_diffusers_subfolder_candidates(json.load(fp))
            )
        for cand in candidate_subfolders:
          if cand not in gcs_candidates:
            gcs_candidates.append(cand)

        for cand in gcs_candidates:
          gcs_rel = (
              f"{name}/{cand}/config.json" if cand else f"{name}/config.json"
          )
          dest_config = self.fetch_gcs_file(gcs_rel)
          if dest_config and dest_config.exists():
            raw_config = auto_model.AutoModel.load_config(
                str(dest_config.parent)
            )
            resolved_subfolder = cand
            break

      if raw_config is None:
        if load_weights:
          raise ValueError(
              f"load_weights cannot be set to True for {name} when model is"
              " not available in cache."
          )
        raise ValueError(
            f"Model config for '{name}' is missing in local resources"
            f" ({self._FILES.joinpath(str(pathlib.Path(name) / 'config.json'))})"
            " and could not be loaded from cache or GCS."
        )

      return module_cache.ConfigCacheEntry(
          config=raw_config,
          path=model_path,
          subfolder=resolved_subfolder,
      )

    entry = module_cache.CONFIG_CACHE.get(
        "diffusers",
        base_path,
        name,
        subfolder,
        load_weights=load_weights,
        loader=_loader,
    )
    return entry.config, entry.subfolder, entry.path

  def get_module_spec(
      self,
      name: str,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Creates a ModuleSpec for a Diffuser model.

    Args:
      name: Name (str) of the hf diffuser model.
      load_weights: If True, loads pretrained weights. If False, initializes
        with random weights.
      modify_config_hook: A callable that accepts and returns a config
        dictionary to modify the model configuration for diffusers models.
      **kwargs: Additional keyword arguments. Supported arguments: - subfolder:
        Subfolder of the model to load.

    Returns:
      A ModuleSpec containing the model factory and input factory.
    """
    subfolder = kwargs.get("subfolder")
    d_type = torch.bfloat16

    raw_config, subfolder, model_path = self._load_config(
        name, subfolder=subfolder, load_weights=load_weights
    )

    if modify_config_hook is not None:
      if load_weights:
        raise NotImplementedError(
            "modify_config_hook is not supported when load_weights is True."
        )
      raw_config = modify_config_hook(raw_config)
    config_dict = dict(raw_config)
    if config_dict.get("encoder_hid_dim_type") == "ip_image_proj":
      config_dict["encoder_hid_dim_type"] = None

    def _module_factory(dtype: torch.dtype | None = None) -> torch.nn.Module:
      cast_dtype = d_type if dtype is None else dtype
      if load_weights:
        return auto_model.AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=cast_dtype,
            subfolder=subfolder,
        )
      else:
        class_name, model_cls = _resolve_diffusers_model_class(
            name, config_dict
        )
        if class_name == "UNet2DConditionModel" or model_cls == getattr(
            diffusers, "UNet2DConditionModel", None
        ):
          if "num_attention_heads" in config_dict:
            head_val = config_dict.pop("num_attention_heads")
            if "attention_head_dim" not in config_dict:
              config_dict["attention_head_dim"] = head_val
        # Without an explicit `dtype`, keep constructing in the default dtype,
        # which keeps the random weights unchanged.
        with _default_dtype(dtype):
          model = model_cls.from_config(config_dict)
        if isinstance(model, tuple):
          model = model[0]
        return model.to(cast_dtype)

    def _input_factory(shape=None, device="cpu"):
      g = torch.Generator(device="cpu").manual_seed(42)
      batch_size = shape[0] if shape else 1
      seq_len = shape[1] if shape and len(shape) > 1 else 77
      cfg = config_dict

      latent_channels = cfg.get("in_channels", 4)

      # Use latent dimensions passed in shape if available
      # (e.g. for video models)
      if shape and len(shape) > 2:
        latent_dims = shape[2:]
      else:
        # Fall back to sample_size from config for 2D models
        latent_size = cfg.get("sample_size", 64)
        if isinstance(latent_size, int):
          latent_dims = (latent_size, latent_size)
        else:
          latent_dims = tuple(latent_size)

        # Check for 3D model if we only have 2D defaults.
        # If patch_size has 3 dimensions, it's likely a 3D/video model and
        # expects a "frames" dimension.
        patch_size = cfg.get("patch_size")
        if (
            len(latent_dims) == 2
            and isinstance(patch_size, (list, tuple))
            and len(patch_size) == 3
        ):
          latent_dims = (2,) + latent_dims  # Default 2 frames

      class_name, model_cls = _resolve_diffusers_model_class(name, cfg)
      forward_params = inspect.signature(model_cls.forward).parameters

      if len(latent_dims) == 2 and (
          class_name in ("UNet3DConditionModel", "I2VGenXLUNet")
          or "3D" in str(class_name)
          or "fps" in forward_params
          or any(
              "3d" in str(b).lower() for b in cfg.get("down_block_types") or []
          )
      ):
        latent_dims = (2,) + latent_dims  # Default 2 frames

      rope_kwargs = {}
      if "img_ids" in forward_params and "txt_ids" in forward_params:
        # Transformer-based diffusion models (Flux, Chroma, Bria) expect
        # 3D flattened patch token sequence [batch_size, seq_len_img,
        # in_channels] rather than 4D spatial tensors [batch_size, in_channels,
        # height, width].
        latent_h = 4
        latent_w = 4
        seq_len_img = latent_h * latent_w
        noisy_latents = torch.randn(
            (batch_size, seq_len_img, latent_channels),
            generator=g,
            dtype=d_type,
        ).to(device)

        axes_dims_rope = cfg.get("axes_dims_rope")
        num_axes = len(axes_dims_rope) if axes_dims_rope else 3
        img_ids = torch.zeros(
            (seq_len_img, num_axes), device=device, dtype=d_type
        )
        txt_ids = torch.zeros((seq_len, num_axes), device=device, dtype=d_type)
        rope_kwargs["img_ids"] = img_ids
        rope_kwargs["txt_ids"] = txt_ids
      elif "img_shapes" in forward_params:
        # Transformer-based models with image shape tuples (e.g. QwenImage).
        latent_h = 4
        latent_w = 4
        seq_len_img = latent_h * latent_w
        noisy_latents = torch.randn(
            (batch_size, seq_len_img, latent_channels),
            generator=g,
            dtype=d_type,
        ).to(device)
        rope_kwargs["img_shapes"] = [[(1, latent_h, latent_w)]] * batch_size
      else:
        noisy_latents = torch.randn(
            (batch_size, latent_channels, *latent_dims),
            generator=g,
            dtype=d_type,
        ).to(device)

      # standard num_train_timesteps for diffusion models is typically 1000
      timesteps = torch.randint(
          0, 1000, (batch_size,), generator=g, dtype=torch.long
      ).to(device)

      # Fallback to encoder_hid_dim / text_dim / joint_attention_dim if
      # cross_attention_dim is not specified in config.
      cross_attention_dim = (
          cfg.get("caption_channels")
          or cfg.get("encoder_hid_dim")
          or cfg.get("cross_attention_dim")
          or cfg.get("joint_attention_dim")
          or cfg.get("text_embed_dim")
          or cfg.get("text_dim")
      )
      if cross_attention_dim is None:
        if (
            class_name == "UNet2DConditionModel"
            or model_cls == getattr(diffusers, "UNet2DConditionModel", None)
            or "unet" in str(class_name).lower()
        ):
          cross_attention_dim = 1280
        else:
          cross_attention_dim = 2048
      if isinstance(cross_attention_dim, (list, tuple)):
        cross_attention_dim = cross_attention_dim[0]

      if class_name == "CogVideoXTransformer3DModel":
        seq_len = cfg.get("max_text_seq_length", 226)
        num_frames = 2
        h = cfg.get("sample_height", 60)
        w = cfg.get("sample_width", 90)
        noisy_latents = torch.randn(
            (batch_size, num_frames, latent_channels, h, w),
            generator=g,
            dtype=d_type,
        ).to(device)

      # Encoder hidden states - These would be per token text embeddings
      # returned by CLIP-ViT/L text encoder
      dummy_encoder_hidden_states = torch.randn(
          (batch_size, seq_len, cross_attention_dim),
          generator=g,
          dtype=d_type,
      ).to(device)

      # Dynamically determine the primary input key
      if "sample" in forward_params:
        primary_input_key = "sample"
      elif "hidden_states" in forward_params:
        primary_input_key = "hidden_states"
      else:
        # Fallback to the first non-'self' argument of the forward method.
        non_self_params = [p for p in forward_params.keys() if p != "self"]
        if non_self_params:
          primary_input_key = non_self_params[0]
        else:
          raise ValueError(
              f"Could not determine primary input key for {name}. No suitable"
              " parameter found in forward signature."
          )

      if class_name == "Cosmos3OmniTransformer":
        p = cfg.get("latent_patch_size", 2)
        c = cfg.get("latent_channel", 48)
        t_frames, h, w = 1, p, p
        t_patches = t_frames
        h_patches = h // p
        w_patches = w // p
        num_patches = t_patches * h_patches * w_patches
        und_len = 4
        seq_len = und_len + num_patches
        input_ids = torch.randint(
            0,
            cfg.get("vocab_size", 151936),
            (und_len,),
            device=device,
            dtype=torch.long,
        )
        text_indexes = torch.arange(und_len, device=device, dtype=torch.long)
        position_ids = torch.zeros(
            (3, seq_len), device=device, dtype=torch.long
        )
        vision_tokens = [
            torch.randn(
                (1, c, t_frames, h, w), generator=g, dtype=d_type, device=device
            )
        ]
        vision_token_shapes = [(t_patches, h_patches, w_patches)]
        vision_sequence_indexes = torch.arange(
            und_len, seq_len, device=device, dtype=torch.long
        )
        vision_mse_loss_indexes = vision_sequence_indexes
        vision_timesteps = torch.tensor([1.0], device=device, dtype=d_type)
        vision_noisy_frame_indexes = [
            torch.tensor([0], device=device, dtype=torch.long)
        ]
        return (), {
            "input_ids": input_ids,
            "text_indexes": text_indexes,
            "position_ids": position_ids,
            "und_len": und_len,
            "sequence_length": seq_len,
            "vision_tokens": vision_tokens,
            "vision_token_shapes": vision_token_shapes,
            "vision_sequence_indexes": vision_sequence_indexes,
            "vision_mse_loss_indexes": vision_mse_loss_indexes,
            "vision_timesteps": vision_timesteps,
            "vision_noisy_frame_indexes": vision_noisy_frame_indexes,
        }

      if "cap_feats" in forward_params and "t" in forward_params:
        cap_feat_dim = (
            cfg.get("cap_feat_dim") or cfg.get("joint_attention_dim") or 2560
        )
        h, w = 16, 16
        z_x = [
            torch.randn(
                (latent_channels, 1, h, w),
                generator=g,
                dtype=d_type,
            ).to(device)
            for _ in range(batch_size)
        ]
        z_cap_feats = [
            torch.randn(
                (seq_len, cap_feat_dim),
                generator=g,
                dtype=d_type,
            ).to(device)
            for _ in range(batch_size)
        ]
        z_t = torch.tensor([0.0], device=device, dtype=d_type).expand(
            batch_size
        )
        kwargs = {
            "x": z_x,
            "t": z_t,
            "cap_feats": z_cap_feats,
        }
      else:
        kwargs = {
            primary_input_key: noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": dummy_encoder_hidden_states,
        }
        kwargs.update(rope_kwargs)

      if "guidance" in forward_params and cfg.get("guidance_embeds", False):
        kwargs["guidance"] = torch.tensor(
            [1.0], device=device, dtype=d_type
        ).expand(batch_size)

      if "pooled_projections" in forward_params:
        proj_dim = (
            cfg.get("pooled_projection_dim")
            or cfg.get("projection_class_embeddings_input_dim")
            or 32
        )
        kwargs["pooled_projections"] = torch.randn(
            (batch_size, proj_dim), generator=g, dtype=d_type
        ).to(device)

      if "additional_t_cond" in forward_params:
        kwargs["additional_t_cond"] = torch.zeros(
            (batch_size,), device=device, dtype=torch.long
        )

      if "proj_embedding" in forward_params:
        emb_dim = cfg.get("embedding_dim", 768)
        num_emb = cfg.get("num_embeddings", 77)
        kwargs["hidden_states"] = torch.randn(
            (batch_size, emb_dim), generator=g, dtype=d_type
        ).to(device)
        kwargs["proj_embedding"] = torch.randn(
            (batch_size, emb_dim), generator=g, dtype=d_type
        ).to(device)
        kwargs["encoder_hidden_states"] = torch.randn(
            (batch_size, num_emb, emb_dim), generator=g, dtype=d_type
        ).to(device)

      if "prior_token_id" in forward_params:
        kwargs["prior_token_id"] = torch.zeros(
            (batch_size,), device=device, dtype=torch.long
        )
        kwargs["prior_token_drop"] = torch.zeros(
            (batch_size,), device=device, dtype=torch.bool
        )
        kwargs["target_size"] = torch.tensor(
            [[1024, 1024]] * batch_size, dtype=torch.float32, device=device
        )
        kwargs["crop_coords"] = torch.tensor(
            [[0, 0]] * batch_size, dtype=torch.float32, device=device
        )

      # Create dummy additional conditioning inputs which would be expected in
      # an SDXL pipeline.
      if cfg.get("addition_embed_type") == "text_time":
        proj_dim = cfg.get("projection_class_embeddings_input_dim", 2816)
        time_dim = cfg.get("addition_time_embed_dim", 256)
        text_embeds_dim = proj_dim - (6 * time_dim)

        # Pooled text embeddings - A single embedding vector per batch
        # representing the entire text sequence returned by OpenCLIP-ViT/G
        dummy_pooled_embeds = torch.randn(
            (batch_size, text_embeds_dim), generator=g, dtype=d_type
        ).to(device)
        # time_ids - Tensor of shape [batch_size, 6] providing crop information
        # to the unet (nothing to do with timesteps).
        # Represents: [orig_height, orig_width, crops_coords_top,
        # crops_coords_left, target_height, target_width]
        # Here, we initialize random values.
        dummy_time_ids = torch.randn(
            (batch_size, 6), generator=g, dtype=d_type
        ).to(device)

        kwargs["added_cond_kwargs"] = {
            "text_embeds": dummy_pooled_embeds,
            "time_ids": dummy_time_ids,
        }
      elif cfg.get("addition_embed_type") == "text_image":
        img_dim = cfg.get("cross_attention_dim") or cross_attention_dim
        if isinstance(img_dim, (list, tuple)):
          img_dim = img_dim[0]
        kwargs["added_cond_kwargs"] = {
            "text_embeds": (
                torch.randn(
                    (batch_size, img_dim), generator=g, dtype=d_type
                ).to(device)
            ),
            "image_embeds": (
                torch.randn(
                    (batch_size, img_dim), generator=g, dtype=d_type
                ).to(device)
            ),
        }
      elif cfg.get("addition_embed_type") == "image":
        img_dim = (
            cfg.get("encoder_hid_dim")
            or cfg.get("cross_attention_dim")
            or cross_attention_dim
        )
        if isinstance(img_dim, (list, tuple)):
          img_dim = img_dim[0]
        kwargs["added_cond_kwargs"] = {
            "image_embeds": (
                torch.randn(
                    (batch_size, img_dim), generator=g, dtype=d_type
                ).to(device)
            ),
        }
      elif cfg.get("addition_embed_type") == "image_hint":
        img_dim = (
            cfg.get("encoder_hid_dim")
            or cfg.get("cross_attention_dim")
            or cross_attention_dim
        )
        if isinstance(img_dim, (list, tuple)):
          img_dim = img_dim[0]
        hint_h = latent_dims[-2] * 8 if len(latent_dims) >= 2 else 512
        hint_w = latent_dims[-1] * 8 if len(latent_dims) >= 1 else 512
        kwargs["added_cond_kwargs"] = {
            "image_embeds": (
                torch.randn(
                    (batch_size, img_dim), generator=g, dtype=d_type
                ).to(device)
            ),
            "hint": (
                torch.randn(
                    (batch_size, 3, hint_h, hint_w), generator=g, dtype=d_type
                ).to(device)
            ),
        }
      elif "added_cond_kwargs" in forward_params:
        if cfg.get("sample_size") == 128 or cfg.get(
            "use_additional_conditions"
        ):
          h = latent_dims[-2] if len(latent_dims) >= 2 else 128
          w = latent_dims[-1] if len(latent_dims) >= 2 else 128
          resolution = torch.tensor(
              [float(h), float(w)], device=device, dtype=d_type
          ).repeat(batch_size, 1)
          aspect_ratio = torch.tensor(
              [float(h / w)], device=device, dtype=d_type
          ).repeat(batch_size, 1)
          kwargs["added_cond_kwargs"] = {
              "resolution": resolution,
              "aspect_ratio": aspect_ratio,
          }
        else:
          kwargs["added_cond_kwargs"] = {
              "resolution": None,
              "aspect_ratio": None,
          }

      if "encoder_hidden_states" not in forward_params:
        kwargs.pop("encoder_hidden_states", None)
      if "added_cond_kwargs" not in forward_params:
        kwargs.pop("added_cond_kwargs", None)
      if "timestep" not in forward_params:
        kwargs.pop("timestep", None)
      if "class_labels" in forward_params:
        if (
            cfg.get("num_class_embeds") is not None
            or cfg.get("class_embed_type") is not None
        ):
          if cfg.get("class_embed_type") == "projection":
            proj_dim = cfg.get("projection_class_embeddings_input_dim", 512)
            kwargs["class_labels"] = torch.zeros(
                (batch_size, proj_dim), dtype=d_type, device=device
            )
          elif cfg.get("class_embed_type") == "timestep":
            kwargs["class_labels"] = torch.zeros(
                (batch_size,), dtype=d_type, device=device
            )
          else:
            kwargs["class_labels"] = torch.zeros(
                (batch_size,), dtype=torch.long, device=device
            )

      if "controlnet_cond" in forward_params:
        if "img_shapes" in forward_params:
          latent_h, latent_w = 4, 4
          seq_len_img = latent_h * latent_w
          extra_channels = cfg.get("extra_condition_channels", 0)
          kwargs["controlnet_cond"] = torch.randn(
              (batch_size, seq_len_img, latent_channels + extra_channels),
              generator=g,
              dtype=d_type,
          ).to(device)
        elif (
            "img_ids" in forward_params
            and "txt_ids" in forward_params
            and cfg.get("conditioning_embedding_channels") is None
        ):
          latent_h, latent_w = 4, 4
          seq_len_img = latent_h * latent_w
          kwargs["controlnet_cond"] = torch.randn(
              (batch_size, seq_len_img, latent_channels),
              generator=g,
              dtype=d_type,
          ).to(device)
        elif class_name == "SD3ControlNetModel":
          patch_size = cfg.get("patch_size", 2)
          latent_h = 16
          latent_w = 16
          cond_channels = cfg.get("in_channels", 16) + cfg.get(
              "extra_conditioning_channels", 0
          )
          kwargs["controlnet_cond"] = torch.randn(
              (batch_size, cond_channels, latent_h, latent_w),
              generator=g,
              dtype=d_type,
          ).to(device)
          if not cfg.get("use_pos_embed", True):
            inner_dim = cfg.get("num_attention_heads", 18) * cfg.get(
                "attention_head_dim", 64
            )
            num_patches = (latent_h // patch_size) * (latent_w // patch_size)
            kwargs[primary_input_key] = torch.randn(
                (batch_size, num_patches, inner_dim),
                generator=g,
                dtype=d_type,
            ).to(device)
          else:
            kwargs[primary_input_key] = torch.randn(
                (batch_size, latent_channels, latent_h, latent_w),
                generator=g,
                dtype=d_type,
            ).to(device)
          if cfg.get("joint_attention_dim") is None:
            kwargs.pop("encoder_hidden_states", None)
        else:
          cond_channels = cfg.get("conditioning_channels", 3)
          out_channels = cfg.get("conditioning_embedding_out_channels") or (
              16,
              32,
              96,
              256,
          )
          factor = 2 ** (len(out_channels) - 1)
          if len(latent_dims) == 2:
            cond_h = latent_dims[0] * factor
            cond_w = latent_dims[1] * factor
          else:
            cond_h, cond_w = 64 * factor, 64 * factor
          kwargs["controlnet_cond"] = torch.randn(
              (batch_size, cond_channels, cond_h, cond_w),
              generator=g,
              dtype=d_type,
          ).to(device)

      if "controlnet_mode" in forward_params:
        kwargs["controlnet_mode"] = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )

      if "fps" in forward_params:
        kwargs["fps"] = torch.tensor(
            [16], device=device, dtype=torch.long
        ).expand(batch_size)
      if "image_latents" in forward_params:
        kwargs["image_latents"] = torch.randn(
            (batch_size, latent_channels, *latent_dims),
            generator=g,
            dtype=d_type,
        ).to(device)
      if "image_embeddings" in forward_params:
        kwargs["image_embeddings"] = torch.randn(
            (batch_size, cross_attention_dim),
            generator=g,
            dtype=d_type,
        ).to(device)

      if (
          "encoder_hidden_states_t5" in forward_params
          and "encoder_hidden_states_llama3" in forward_params
      ):
        caption_channels = cfg.get("caption_channels") or [4096, 2048]
        t5_dim = caption_channels[0]
        llama_dim = (
            caption_channels[1]
            if len(caption_channels) > 1
            else caption_channels[0]
        )
        llama_layers = cfg.get("llama_layers") or [0, 1]
        num_llama_layers = max(llama_layers) + 1
        text_emb_dim = cfg.get("text_emb_dim") or 2048

        kwargs["encoder_hidden_states_t5"] = torch.randn(
            (batch_size, seq_len, t5_dim), generator=g, dtype=d_type
        ).to(device)
        kwargs["encoder_hidden_states_llama3"] = torch.randn(
            (num_llama_layers, batch_size, seq_len, llama_dim),
            generator=g,
            dtype=d_type,
        ).to(device)
        kwargs["pooled_embeds"] = torch.randn(
            (batch_size, text_emb_dim), generator=g, dtype=d_type
        ).to(device)
        if "timesteps" in forward_params:
          kwargs["timesteps"] = timesteps
        kwargs.pop("encoder_hidden_states", None)

      if (
          "encoder_hidden_states_t5" in forward_params
          and "image_meta_size" in forward_params
      ):
        text_len = cfg.get("text_len", 77)
        text_len_t5 = cfg.get("text_len_t5", 256)
        cross_attention_dim = cfg.get("cross_attention_dim", 1024)
        cross_attention_dim_t5 = cfg.get("cross_attention_dim_t5", 2048)

        kwargs["encoder_hidden_states"] = torch.randn(
            (batch_size, text_len, cross_attention_dim),
            generator=g,
            dtype=d_type,
        ).to(device)
        kwargs["text_embedding_mask"] = torch.ones(
            (batch_size, text_len), device=device, dtype=torch.bool
        )
        kwargs["encoder_hidden_states_t5"] = torch.randn(
            (batch_size, text_len_t5, cross_attention_dim_t5),
            generator=g,
            dtype=d_type,
        ).to(device)
        kwargs["text_embedding_mask_t5"] = torch.ones(
            (batch_size, text_len_t5), device=device, dtype=torch.bool
        )
        kwargs["image_meta_size"] = torch.tensor(
            [[1024.0, 1024.0, 1024.0, 1024.0, 0.0, 0.0]],
            device=device,
            dtype=d_type,
        ).repeat(batch_size, 1)
        if "style" in forward_params:
          kwargs["style"] = torch.zeros(
              (batch_size,), device=device, dtype=torch.long
          )
        if "timestep" in forward_params:
          kwargs["timestep"] = timesteps.to(dtype=d_type)
        if "controlnet_cond" in forward_params:
          kwargs["controlnet_cond"] = torch.randn(
              (batch_size, latent_channels, *latent_dims),
              generator=g,
              dtype=d_type,
          ).to(device)

      if "text_bth" in forward_params and "text_lens" in forward_params:
        text_in_dim = (
            cfg.get("text_in_dim")
            or cfg.get("hidden_size")
            or cross_attention_dim
        )
        kwargs["text_bth"] = torch.randn(
            (batch_size, seq_len, text_in_dim),
            generator=g,
            dtype=d_type,
        ).to(device)
        kwargs["text_lens"] = torch.tensor(
            [seq_len] * batch_size, device=device, dtype=torch.long
        )
        kwargs.pop("encoder_hidden_states", None)

      if "text_encoder_layers" in forward_params:
        num_layers = (cfg.get("num_layers") or 19) + (
            cfg.get("num_single_layers") or 38
        )
        text_encoder_dim = cfg.get("text_encoder_dim") or 2048
        kwargs["text_encoder_layers"] = [
            torch.randn(
                (batch_size, seq_len, text_encoder_dim),
                generator=g,
                dtype=d_type,
            ).to(device)
            for _ in range(num_layers)
        ]

      return (), kwargs

    return ModuleSpec(
        _module_factory,
        _input_factory,
        config=config_dict,
        modality=Modality.DIFFUSION,
    )


class ModuleRegistry:
  """Central registry for managing multiple model providers."""

  def __init__(self, base_path: str | None = None):
    """Initializes the registry.

    Providers construct specific paths relative to this base_path.
    - For Timm and Torchvision library, models are structured as:
    base_path/{provider}/{model}/
    - For other libraries, models are structured as:
    base_path/{provider}/{owner}/{model}/

    Args:
      base_path: The base directory for model weights. Defaults to the value of
        `_WEIGHTS_BASE_PATH`.
    """
    if base_path is None:
      base_path = _WEIGHTS_BASE_PATH.value

    self._providers: dict[str, BaseProvider] = {}
    if _HAS_TORCHVISION:
      self._providers["torchvision"] = TorchvisionProvider(base_path=base_path)
    if _HAS_TIMM:
      self._providers["timm"] = TimmProvider(base_path=base_path)
    if _HAS_TRANSFORMERS:
      self._providers["transformers"] = TransformersProvider(
          base_path=base_path
      )
    if _HAS_DIFFUSERS:
      self._providers["diffusers"] = DiffusersProvider(base_path=base_path)

  def list_all_sources(self) -> Sequence[str]:
    """Returns a list of registered provider keys (e.g., 'torchvision')."""
    return list(self._providers.keys())

  def list_all_modules(self) -> Sequence[str]:
    """Returns a unified list of all available modules across all providers.

    Returns:
      A list of strings formatted as '{source}/{model_name}'.
    """
    modules = []
    for key, p in self._providers.items():
      for m in p.list_modules():
        modules.append(f"{key}/{m}")
    return modules

  def _get_provider(self, source: str) -> BaseProvider:
    canonical_source = _PROVIDER_ALIASES.get(source, source)
    if canonical_source not in self._providers:
      raise ValueError(f"Source '{source}' not supported.")
    return self._providers[canonical_source]

  def list_modules(self, source: str) -> Sequence[str]:
    """Lists available models for a specific source.

    Args:
      source: The provider key (e.g., 'timm', 'transformers').

    Returns:
      A list of model names.

    Raises:
      ValueError: If the source is not found in the registry.
    """
    return self._get_provider(source).list_modules()

  def get_module_spec(
      self,
      source: str,
      name: str | None = None,
      *,
      load_weights: bool = False,
      modify_config_hook: Callable[[Any], Any] | None = None,
      **kwargs,
  ) -> ModuleSpec:
    """Instantiates and returns the ModuleSpec for a specific model.

    Args:
      source: The provider key (e.g., 'timm', 'transformers'), or a
        fully-qualified module name formatted as '{source}/{model_name}'.
      name: The name of the model within that provider. If omitted, `source` is
        treated as a qualified name ('{source}/{model_name}').
      load_weights: Whether to load pre-trained weights.
      modify_config_hook: A callable to modify the model configuration. The
        callable accepts and returns a config object specific to the model
        library (e.g., transformers.PretrainedConfig for transformers).
      **kwargs: Additional provider-specific arguments.

    Returns:
      A ModuleSpec containing the model factory and input factory.

    Raises:
      ValueError: If the source is not found in the registry, or if name is
        omitted and source does not contain a provider prefix.
    """
    if name is None:
      if "/" not in source:
        raise ValueError(
            "When 'name' is omitted, 'source' must be formatted as"
            f" '{{source}}/{{model_name}}', got: {source!r}"
        )
      source, _, name = source.partition("/")

    canonical_source = _PROVIDER_ALIASES.get(source, source)
    for prefix in (f"{source}/", f"{canonical_source}/"):
      if name.startswith(prefix):
        name = name.removeprefix(prefix)
        break

    provider = self._get_provider(canonical_source)
    return provider.get_module_spec(
        name,
        load_weights=load_weights,
        modify_config_hook=modify_config_hook,
        **kwargs,
    )
