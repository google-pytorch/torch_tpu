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

"""Provides a unified registry for instantiating PyTorch models from various sources.

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
from collections.abc import Callable, Mapping, Sequence
import copy
import enum
import functools
from importlib import resources
from importlib.resources import abc as resources_abc
import inspect
import json
import os
import pathlib
import tempfile
import threading
import types
from typing import Any, Iterator, NamedTuple, overload
import uuid

from absl import flags
from absl import logging
from etils import epath
from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage
from safetensors import torch as safetensors_torch
import torch

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

_AUDIO_MODEL_TYPES = (
    # go/keep-sorted start
    "audio",
    "audio-flamingo",
    "audio-spectrogram-transformer",
    "audio_flamingo",
    "bark",
    "clap",
    "csm",
    "dac",
    "data2vec-audio",
    "dia",
    "encodec",
    "fastspeech",
    "fastspeech2",
    "gemma3naudio",
    "gemma4audio",
    "granite-speech",
    "granite_speech",
    "hubert",
    "lasr_ctc",
    "mimi",
    "moonshine",
    "moonshine_streaming",
    "moshi",
    "musicgen",
    "omnitoken2wav",
    "parakeet_ctc",
    "parakeet_encoder",
    "parakeet_rnnt",
    "pe-a-frame",
    "pe-audio",
    "pe_audio",
    "pe_audio_encoder",
    "pop2piano",
    "qwen2-audio",
    "qwen2_audio",
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
    "efficientloftr",
    "efficientnet",
    "eomt",
    "focalnet",
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
    "tvp",
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
    "qformer",
    "qwen2-audio",
    "qwen2_5_vl",
    "qwen2_audio",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "sam3",
    "sam3_lite_text",
    "siglip",
    "smolvlm",
    "udop",
    "vilt",
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
    blob.download_to_filename(str(temp_dest))
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


_CacheKey = tuple[str, str | None, str, str | None, bool]


class _ConfigCacheEntry(NamedTuple):
  config: Any
  path: epath.Path | None = None
  subfolder: str | None = None


class _ConfigCacheManager:
  """Thread-safe in-memory cache manager for model configurations across providers."""

  def __init__(self):
    self._global_lock = threading.Lock()
    self._key_locks: dict[_CacheKey, threading.Lock] = {}
    self._cache: dict[_CacheKey, _ConfigCacheEntry] = {}

  def __len__(self) -> int:
    with self._global_lock:
      return len(self._cache)

  def _values(self) -> list[_ConfigCacheEntry]:
    with self._global_lock:
      return list(self._cache.values())

  def clear(self) -> None:
    """Clears all cached configurations."""
    with self._global_lock:
      self._cache.clear()

  def _make_key(
      self,
      provider: str,
      base_path: epath.Path | str | None,
      name: str,
      subfolder: str | None = None,
      load_weights: bool = False,
  ) -> _CacheKey:
    return (
        provider,
        str(base_path) if base_path is not None else None,
        name,
        subfolder,
        load_weights,
    )

  def _get_key_lock(self, key: _CacheKey) -> threading.Lock:
    with self._global_lock:
      if key not in self._key_locks:
        self._key_locks[key] = threading.Lock()
      return self._key_locks[key]

  def get(
      self,
      provider: str,
      base_path: epath.Path | str | None,
      name: str,
      subfolder: str | None = None,
      *,
      load_weights: bool = False,
      loader: Callable[[], _ConfigCacheEntry],
  ) -> _ConfigCacheEntry:
    """Gets cached entry or loads and stores it atomically per key."""
    key = self._make_key(
        provider, base_path, name, subfolder, load_weights=load_weights
    )

    # 1. Fast-path lookup
    with self._global_lock:
      cached = self._cache.get(key)
    if cached is not None:
      return copy.deepcopy(cached)

    # 2. Acquire per-key lock
    key_lock = self._get_key_lock(key)
    with key_lock:
      # 3. Double-check cache inside per-key lock
      with self._global_lock:
        cached = self._cache.get(key)
      if cached is not None:
        return copy.deepcopy(cached)

      # 4. Load fresh entry outside global lock
      fresh_entry = loader()

      # 5. Store defensive deep-copy in cache
      cached_entry = copy.deepcopy(fresh_entry)
      with self._global_lock:
        self._cache[key] = cached_entry

      return fresh_entry


_CONFIG_CACHE = _ConfigCacheManager()


_PreprocessorCacheKey = tuple[Any, ...]


class _PreprocessorCacheManager:
  """Thread-safe in-memory cache manager for model preprocessors and tokenizers."""

  def __init__(self):
    self._global_lock = threading.Lock()
    self._key_locks: dict[_PreprocessorCacheKey, threading.Lock] = {}
    self._cache: dict[_PreprocessorCacheKey, Any] = {}

  def __len__(self) -> int:
    with self._global_lock:
      return len(self._cache)

  def clear(self) -> None:
    """Clears all cached preprocessors and tokenizers."""
    with self._global_lock:
      self._cache.clear()

  def _get_key_lock(self, key: _PreprocessorCacheKey) -> threading.Lock:
    with self._global_lock:
      if key not in self._key_locks:
        self._key_locks[key] = threading.Lock()
      return self._key_locks[key]

  def get_or_load(
      self,
      key: _PreprocessorCacheKey,
      loader: Callable[[], Any],
  ) -> Any:
    """Gets cached preprocessor or loads and stores it atomically per key."""
    with self._global_lock:
      if key in self._cache:
        return self._cache[key]

    key_lock = self._get_key_lock(key)
    with key_lock:
      with self._global_lock:
        if key in self._cache:
          return self._cache[key]

      fresh_entry = loader()

      with self._global_lock:
        self._cache[key] = fresh_entry

      return fresh_entry


_PREPROCESSOR_CACHE = _PreprocessorCacheManager()


def clear_preprocessor_cache() -> None:
  """Clears the in-memory preprocessor and tokenizer cache."""
  _PREPROCESSOR_CACHE.clear()


def clear_config_cache() -> None:
  """Clears the in-memory config and preprocessor caches for all providers."""
  _CONFIG_CACHE.clear()
  _PREPROCESSOR_CACHE.clear()


class _ConfigWrapper:
  """Wrapper around HF config dictionaries and objects for consistent attribute and dict access."""

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

  def __setstate__(self, state: dict[str, Any]) -> None:
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
    if hasattr(data, "allow_global_per_layer_attribute_access"):
      try:
        data.allow_global_per_layer_attribute_access = True
      except Exception:
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
    if isinstance(self._data, dict):
      self._data[name] = value
    else:
      setattr(self._data, name, value)

  def get(self, key: str, default: Any = None) -> Any:
    data = self._data
    if hasattr(data, "allow_global_per_layer_attribute_access"):
      try:
        data.allow_global_per_layer_attribute_access = True
      except Exception:
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
      except Exception:
        val = default
    if val is default:
      return default
    if val is not None and default is not None:
      try:
        val = type(default)(val)
      except Exception:
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
  """Wraps a config dictionary or object in _ConfigWrapper if not already wrapped."""
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


class ModuleSpec:
  """A specification container for a PyTorch model and its inputs.

  Attributes:
    module_factory: A callable that returns an instantiated `torch.nn.Module`.
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
      module_factory: Callable[[], torch.nn.Module],
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
      if not self._base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {self._base_path}")
    else:
      self._base_path = None

  @property
  def _cloud_bucket_path(self) -> str | None:
    """Returns the GCS blob prefix for this provider, or None if GCS is not configured."""
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
    """Fetches a file from GCS, using the local temp cache if already downloaded.

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
  def list_modules(self) -> list[str]:
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


class TorchvisionProvider(BaseProvider):
  """Provider for standard Torchvision models."""

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="torchvision")

  def list_modules(self) -> list[str]:
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

    def _module_factory():
      if load_weights and self._base_path:
        local_checkpoint = None
        local_dir = self._base_path / name

        if local_dir.exists() and local_dir.is_dir():
          for p in local_dir.iterdir():
            if p.suffix in (".pt", ".pth", ".bin", ".safetensors"):
              local_checkpoint = p
              break

        if local_checkpoint and local_checkpoint.exists():
          try:
            model = torchvision.models.get_model(
                name, weights=None, weights_backbone=None
            )
          except TypeError:
            model = torchvision.models.get_model(name, weights=None)

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
      try:
        return torchvision.models.get_model(
            name, weights=None, weights_backbone=None
        )
      except TypeError:
        return torchvision.models.get_model(name, weights=None)

    return ModuleSpec(
        _module_factory,
        _torchvision_input_factory,
        modality=Modality.VISION,
    )


class TimmProvider(BaseProvider):
  """Provider for TIMM (PyTorch Image Models)."""

  def __init__(self, base_path: str | None = None):
    super().__init__(base_path=base_path, subdir="timm")

  def list_modules(self) -> list[str]:
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

    try:
      config = timm.models.get_pretrained_cfg(name)
    except (RuntimeError, ValueError):
      config = None
      logging.warning(
          "Couldn't find config for %s.",
          name,
      )

    if config and modify_config_hook is not None:
      config = modify_config_hook(config)

    def _module_factory():
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

        model = timm.create_model(name, pretrained=False)
        model.load_state_dict(state_dict)
        return model

      return timm.create_model(name, pretrained=False)

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
        _module_factory,
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
  text_cfg = cfg.get("text_config")
  if text_cfg is not None:
    configs_to_check.append(text_cfg)

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
        return val

  # Fallback for relative position models (T5, etc.) or missing data.
  return default


def _walk_package_resources(
    traversable: resources_abc.Traversable,
) -> Iterator[resources_abc.Traversable]:
  """Recursively yields Traversable objects for all files in a directory/package."""
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
  if any(
      k in arch_name
      for k in (
          "imageclassification",
          "imagesegmentation",
          "depthestimation",
          "objectdetection",
      )
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
  """Extracts image_size as an integer from config or vision_config or encoder."""
  cfg = _wrap_config(config)
  vision_config = (
      cfg.get("vision_config")
      or cfg.get("encoder")
      or cfg.get("backbone_config")
  )
  raw_val = vision_config.get("image_size") if vision_config else None
  if isinstance(raw_val, (list, tuple)) and raw_val:
    raw_val = raw_val[0]
  val = _safe_int(raw_val, default=None, min_val=1)
  if val is None:
    raw_val = cfg.get("image_size")
    if isinstance(raw_val, (list, tuple)) and raw_val:
      raw_val = raw_val[0]
    val = _safe_int(raw_val, default=default_size, min_val=1)

  return val


def _get_num_channels(config: Any, default_channels: int = 3) -> int:
  """Extracts num_channels from config or vision_config or encoder."""
  cfg = _wrap_config(config)
  vision_config = (
      cfg.get("vision_config")
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


def _generate_gemma4_inputs(
    config: Any,
    batch_size: int,
    image_size: int,
    device: str,
    input_kwargs: dict[str, Any],
    generator: torch.Generator | None = None,
) -> None:
  """Generates gemma4 specific inputs (patchified image and position ids).

  Unlike standard Vision Transformers that take raw images of shape (B, C, H, W)
  and perform patchification internally (e.g. via Conv2d), Gemma 4 expects
  pre-patchified images from the image processor.

  Specifically:
  - pixel_values: Shape (B, max_patches, patch_pixels) where patch_pixels is
    C * patch_size * patch_size (e.g., 3 * 16 * 16 = 768). Each patch is
    flattened.
  - image_position_ids: Shape (B, max_patches, 2) containing the (x, y) grid
    coordinates of each patch in the original image. This is needed because
    the flattened patch sequence loses spatial structure. Padding patches
    are indicated by (-1, -1).

  It also expects input_ids to contain placeholders (image_token_id) for the
  pooled image tokens, which will be replaced by the vision features in the
  model.

  Args:
    config: The model configuration.
    batch_size: The batch size.
    image_size: The size of the input image.
    device: The device to place the tensors on.
    input_kwargs: The dictionary to populate with the generated inputs.
  """
  cfg = _wrap_config(config)
  vision_config = cfg.get("vision_config")
  patch_size = vision_config.get("patch_size", 16) if vision_config else 16
  pooling_kernel_size = (
      vision_config.get("pooling_kernel_size", 3) if vision_config else 3
  )

  # Default max_soft_tokens from Gemma4ImageProcessor.
  # This is the budget of soft tokens for the model.
  max_soft_tokens = 280
  max_patches = max_soft_tokens * pooling_kernel_size**2

  # Determine real patches based on image_size.
  grid_size = image_size // patch_size
  num_real_patches = grid_size * grid_size

  # If the dummy image size results in more patches than the budget,
  # we cap it to the budget to simulate the image processor behavior.
  if num_real_patches > max_patches:
    num_real_patches = max_patches
    grid_size = int(num_real_patches**0.5)
    num_real_patches = grid_size * grid_size

  # Generate pixel_values: (batch_size, max_patches, patch_pixels).
  # Unlike standard ViT models that take raw images and do patchification
  # in the model (e.g. via Conv2d), Gemma 4 expects patchified inputs.
  num_channels = (
      cfg.get("num_channels")
      or (vision_config.get("num_channels") if vision_config else None)
      or 3
  )
  patch_pixels = num_channels * patch_size * patch_size

  input_kwargs["pixel_values"] = torch.randn(
      batch_size, max_patches, patch_pixels, generator=generator
  ).to(device)

  # Generate image_position_ids: (batch_size, max_patches, 2).
  # We initialize with -1 (padding).
  image_position_ids = torch.full(
      (batch_size, max_patches, 2), -1, device=device, dtype=torch.long
  )

  # Fill in real positions (grid coordinates).
  grid_x, grid_y = torch.meshgrid(
      torch.arange(grid_size, device=device),
      torch.arange(grid_size, device=device),
      indexing="ij",
  )
  coords = torch.stack([grid_x, grid_y], dim=-1)  # (grid_size, grid_size, 2)
  coords = coords.view(-1, 2)  # (grid_size^2, 2)

  # Copy coordinates to the valid part of image_position_ids.
  image_position_ids[:, :num_real_patches, :] = coords.unsqueeze(0).expand(
      batch_size, -1, -1
  )

  input_kwargs["image_position_ids"] = image_position_ids

  # Calculate number of pooled features mathematically.
  # The model pools patches using avg pooling with kernel size
  # pooling_kernel_size.
  # We need to know how many valid features will remain after pooling to
  # insert the correct number of placeholders in input_ids.
  pooled_dim = grid_size // pooling_kernel_size
  num_features = pooled_dim * pooled_dim

  # Overwrite input_ids to have num_features copies of image_token_id.
  # The model expects to find these placeholders to merge text and image
  # features.
  image_token_id = cfg.get("image_token_id", 258880)
  input_kwargs["input_ids"][:, :num_features] = image_token_id


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

  if isinstance(dtype, str):
    return getattr(torch, dtype, torch.float32)
  elif isinstance(dtype, torch.dtype):
    return dtype
  return torch.float32


def _load_transformers_preprocessor(
    model_dir_or_repo_id: epath.PathLike | str | None,
    modality: Modality | None = None,
) -> Any:
  """Loads and caches a HuggingFace processor or tokenizer for a model path."""
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
    if prefer_tokenizer:
      loaders = (
          transformers.AutoTokenizer.from_pretrained,
          transformers.AutoProcessor.from_pretrained,
      )
    else:
      loaders = (
          transformers.AutoProcessor.from_pretrained,
          transformers.AutoTokenizer.from_pretrained,
      )
    for load_fn in loaders:
      try:
        return load_fn(model_path_str)
      except Exception:  # pylint: disable=broad-except
        continue
    return None

  return _PREPROCESSOR_CACHE.get_or_load(cache_key, _loader)


def _sanitize_transformers_config_dict(
    config_dict: dict[str, Any],
    load_weights: bool = False,
) -> dict[str, Any]:
  """Sanitizes HuggingFace config dictionary to fix common schema and offline issues."""
  config_dict = dict(config_dict)

  # Convert id2label values to strings
  if "id2label" in config_dict and isinstance(config_dict["id2label"], dict):
    config_dict["id2label"] = {
        str(k): str(v) for k, v in config_dict["id2label"].items()
    }

  # Convert int dropout probabilities to floats
  for float_key in (
      "hidden_dropout_prob",
      "attention_probs_dropout_prob",
      "classifier_dropout",
  ):
    val = config_dict.get(float_key)
    if isinstance(val, int) and not isinstance(val, bool):
      config_dict[float_key] = float(val)

  # For offline execution, disable downloading pretrained backbones.
  if not load_weights and "use_pretrained_backbone" in config_dict:
    config_dict["use_pretrained_backbone"] = False

  # For offline execution, synthesize backbone_config if missing to prevent network calls.
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
      config_dict["backbone_config"] = {
          "model_type": "resnet",
      }

  return config_dict


def _load_transformers_config_with_fallback(
    model_path_or_file: epath.PathLike | str,
    load_weights: bool = False,
) -> Any:
  """Loads AutoConfig with sanitization fallback for schema inconsistencies and offline constraints."""
  # 1. Standard AutoConfig.from_pretrained with offline kwargs
  kwargs = {}
  if not load_weights:
    kwargs["use_pretrained_backbone"] = False
  try:
    return _wrap_config(
        transformers.AutoConfig.from_pretrained(
            str(model_path_or_file), **kwargs
        )
    )
  except Exception:  # pylint: disable=broad-except
    pass

  # 2. Read config.json, sanitize schema, and instantiate AutoConfig
  path = epath.Path(model_path_or_file)
  cfg_file = path / "config.json" if path.is_dir() else path
  with cfg_file.open("r", encoding="utf-8") as f:
    config_dict = json.load(f)

  config_dict = _sanitize_transformers_config_dict(
      config_dict, load_weights=load_weights
  )

  model_type = config_dict.get("model_type")
  if not isinstance(model_type, str):
    raise ValueError(
        f"Missing or invalid model_type in config for {model_path_or_file}"
    )

  config_mapping = getattr(
      getattr(transformers, "models", None)
      and getattr(transformers.models, "auto", None)
      and getattr(transformers.models.auto, "configuration_auto", None),
      "CONFIG_MAPPING",
      {},
  )
  if model_type in config_mapping:
    try:
      return _wrap_config(config_mapping[model_type].from_dict(config_dict))
    except Exception:  # pylint: disable=broad-except
      pass
  try:
    return _wrap_config(
        transformers.AutoConfig.for_model(model_type, **config_dict)
    )
  except Exception:  # pylint: disable=broad-except
    return _wrap_config(config_dict)


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

  def list_modules(self) -> list[str]:
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

  def _load_config_and_model_dir(
      self, name: str, load_weights: bool = False
  ) -> tuple[Any, epath.Path | None]:
    base_path = self._base_path if self.has_cache_dir else None

    def _loader() -> _ConfigCacheEntry:
      config = None
      model_dir_or_repo_id: epath.Path | None = None

      if self.has_cache_dir:
        model_dir_or_repo_id = self._base_path / name  # pyrefly: ignore[unsupported-operation]
        try:
          if model_dir_or_repo_id.exists():
            config = _load_transformers_config_with_fallback(
                str(model_dir_or_repo_id), load_weights=load_weights
            )
        except Exception as exc:  # pylint: disable=broad-except
          logging.warning(
              "Failed to access %s in cache, falling back to local resources."
              " Error: %s",
              model_dir_or_repo_id,
              exc,
          )

      if config is None:
        if load_weights:
          raise ValueError(
              f"load_weights cannot be set to True for {name} when falling back"
              " to local configuration resources or GCS."
          )

        # Fallback to local resources
        for files_root in self._FILES:
          try:
            with resources.as_file(
                files_root.joinpath(str(pathlib.Path(name) / "config.json"))
            ) as f:
              if f.exists():
                config = _load_transformers_config_with_fallback(
                    str(f), load_weights=load_weights
                )
                break
          except Exception:  # pylint: disable=broad-except
            config = None

      if config is None:  # Fallback to downloading from GCS
        dest_config = self.fetch_gcs_file(f"{name}/config.json")
        if dest_config and dest_config.exists():
          try:
            config = _load_transformers_config_with_fallback(
                str(dest_config.parent), load_weights=load_weights
            )
            model_dir_or_repo_id = epath.Path(dest_config.parent)
          except Exception as exc:  # pylint: disable=broad-except
            logging.warning(
                "Failed to load GCS config from %s: %s", dest_config.parent, exc
            )
            config = None

      if config is None:
        files_str = ", ".join(
            str(r.joinpath(str(pathlib.Path(name) / "config.json")))
            for r in self._FILES
        )
        raise ValueError(
            f"Model config for '{name}' is missing in local resources"
            f" ({files_str})"
            " and could not be loaded from cache or GCS."
        )

      return _ConfigCacheEntry(
          config=config,
          path=model_dir_or_repo_id,
      )

    entry = _CONFIG_CACHE.get(
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

    Automatically detects the architecture from the config and prepares
    appropriate inputs based on its modality.

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
    # Load the config first
    config, model_dir_or_repo_id = self._load_config_and_model_dir(
        name, load_weights=load_weights
    )

    if modify_config_hook is not None:
      if load_weights:
        raise NotImplementedError(
            "modify_config_hook is not supported when load_weights is True."
        )
      config = _wrap_config(modify_config_hook(config))

    # When loading models offline (load_weights=False), some vision models (like
    # DETR) will still try to download pretrained backbones (e.g. via TIMM)
    # over the network. Disabling this forces offline initialization with
    # random weights, avoiding network errors on Forge.
    if not load_weights and "use_pretrained_backbone" in config:
      config.use_pretrained_backbone = False

    if not load_weights and config.get("model_type", "") == "vits":
      config.use_stochastic_duration_prediction = False

    modality = _determine_modality(config)

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

      target_dtype = _extract_target_dtype(config)
      kwargs.setdefault("low_cpu_mem_usage", True)
      kwargs.setdefault("ignore_mismatched_sizes", True)
      if target_dtype != torch.float32:
        kwargs.setdefault("torch_dtype", target_dtype)
      model_fn = lambda: model_cls.from_pretrained(
          str(model_dir_or_repo_id), **kwargs
      )
    else:
      architectures = config.get("architectures", [])
      model_cls = None

      if architectures:
        arch_name = architectures[0]
        if hasattr(transformers, arch_name):
          try:
            model_cls = getattr(transformers, arch_name)
          except Exception:  # pylint: disable=broad-except
            model_cls = None
        if model_cls is None:
          # Try case-insensitive matching against transformers exports
          arch_lower = arch_name.lower()
          for attr in dir(transformers):
            if attr.lower() == arch_lower:
              try:
                model_cls = getattr(transformers, attr)
                break
              except Exception:  # pylint: disable=broad-except
                continue

      if model_cls is None:
        if modality == Modality.CAUSAL_LM:
          model_cls = transformers.AutoModelForCausalLM
        elif modality == Modality.SEQ2SEQ:
          if "whisper" in config.get("model_type", ""):
            model_cls = getattr(
                transformers,
                "AutoModelForSpeechSeq2Seq",
                transformers.AutoModel,
            )
          else:
            model_cls = transformers.AutoModelForSeq2SeqLM
        else:
          model_cls = transformers.AutoModel

      target_dtype = _extract_target_dtype(config)

      def _create_model():
        cfg = config.unwrap() if isinstance(config, _ConfigWrapper) else config
        m = None
        if model_cls is not None and hasattr(model_cls, "from_config"):
          try:
            m = model_cls.from_config(cfg)
          except Exception as exc:  # pylint: disable=broad-except
            logging.warning(
                "Failed to instantiate %s with from_config: %s. Falling back to"
                " AutoModel.",
                model_cls,
                exc,
            )
        if m is None:
          try:
            if callable(model_cls):
              m = model_cls(cfg)
            else:
              m = transformers.AutoModel.from_config(cfg)
          except Exception:
            m = transformers.AutoModel.from_config(cfg)
        if target_dtype != torch.float32:
          m = m.to(dtype=target_dtype)
        if config.get("model_type", "") in ("vits", "tapas"):
          # Zero TPU float weights before forward pass to prevent uninitialized
          # parameters (from to_empty) causing NaNs/overflows.
          def _zero_weights_pre_hook(module, *_unused_args, **_unused_kwargs):
            for p in module.parameters():
              if p.device.type != "meta" and p.dtype.is_floating_point:
                p.data.zero_()

          m.register_forward_pre_hook(_zero_weights_pre_hook)
        return m

      model_fn = _create_model

    preprocessor_fn = lambda: _load_transformers_preprocessor(
        model_dir_or_repo_id, modality=modality
    )

    def _input_fn(
        shape=None, device="cpu"
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      model_dir = (
          str(model_dir_or_repo_id)
          if model_dir_or_repo_id is not None and model_dir_or_repo_id.exists()
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
    index_dict: dict[str, Any],
) -> list[str]:
  """Extracts prioritized candidate subfolders from a diffusers model_index.json."""
  candidates = []
  for key in _DIFFUSERS_SUBFOLDER_PRIORITY:
    val = index_dict.get(key)
    if val and isinstance(val, (list, tuple)) and val[0] is not None:
      candidates.append(key)
      break
  return candidates


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

  def list_modules(self) -> list[str]:
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
  ) -> tuple[Any, str | None, epath.Path | None]:
    base_path = self._base_path if self.has_cache_dir else None

    def _loader() -> _ConfigCacheEntry:
      raw_config = None
      resolved_subfolder = subfolder
      model_path = None

      # Candidate subfolders for diffusers pipelines and standalone models
      if subfolder:
        candidate_subfolders = [subfolder]
      else:
        candidate_subfolders = []
        if self.has_cache_dir:
          model_path = self._base_path / name  # pyrefly: ignore[unsupported-operation]
          if model_path.exists():
            index_path = model_path / "model_index.json"
            if index_path.exists():
              try:
                with open(index_path, "r") as fp:
                  candidate_subfolders.extend(
                      _extract_diffusers_subfolder_candidates(json.load(fp))
                  )
              except Exception:  # pylint: disable=broad-except
                pass
            # Support nested candidate subfolders
            try:
              for sub_dir in model_path.iterdir():
                if sub_dir.is_dir():
                  for sub_target in ("transformer", "unet"):
                    if (sub_dir / sub_target / "config.json").exists():
                      nested_cand = f"{sub_dir.name}/{sub_target}"
                      if nested_cand not in candidate_subfolders:
                        candidate_subfolders.append(nested_cand)
            except Exception:  # pylint: disable=broad-except
              pass
        for cand in _DEFAULT_DIFFUSERS_SUBFOLDERS:
          if cand not in candidate_subfolders:
            candidate_subfolders.append(cand)

      if self.has_cache_dir:
        model_path = self._base_path / name  # pyrefly: ignore[unsupported-operation]
        try:
          if model_path.exists():
            for cand in candidate_subfolders:
              try:
                raw_config = auto_model.AutoModel.load_config(
                    str(model_path), subfolder=cand
                )
                resolved_subfolder = cand
                break
              except Exception:  # pylint: disable=broad-except
                pass
        except Exception as exc:  # pylint: disable=broad-except
          logging.warning(
              "Failed to access %s in cache, falling back to local resources."
              " Error: %s",
              model_path,
              exc,
          )

      if raw_config is None and not load_weights:  # Fallback to local resources
        model_dir = pathlib.Path(name)
        local_candidates = list(candidate_subfolders)
        index_file = self._FILES.joinpath(str(model_dir / "model_index.json"))
        try:
          with resources.as_file(index_file) as f:
            if f.is_file():
              with open(f, "r") as fp:
                for key in _extract_diffusers_subfolder_candidates(
                    json.load(fp)
                ):
                  if key not in local_candidates:
                    local_candidates.insert(0, key)
        except Exception:  # pylint: disable=broad-except
          pass

        for cand in local_candidates:
          target_rel = (
              model_dir / cand / "config.json"
              if cand
              else model_dir / "config.json"
          )
          try:
            with resources.as_file(self._FILES.joinpath(str(target_rel))) as f:
              if f.is_file():
                raw_config = auto_model.AutoModel.load_config(str(f.parent))
                resolved_subfolder = cand
                break
          except Exception:  # pylint: disable=broad-except
            pass

      if (
          raw_config is None and not load_weights
      ):  # Fallback to downloading from GCS
        gcs_candidates = []
        dest_index = self.fetch_gcs_file(f"{name}/model_index.json")
        if dest_index and dest_index.exists():
          try:
            with open(dest_index, "r") as fp:
              gcs_candidates.extend(
                  _extract_diffusers_subfolder_candidates(json.load(fp))
              )
          except Exception:  # pylint: disable=broad-except
            pass
        for cand in candidate_subfolders:
          if cand not in gcs_candidates:
            gcs_candidates.append(cand)

        for cand in gcs_candidates:
          gcs_rel = (
              f"{name}/{cand}/config.json" if cand else f"{name}/config.json"
          )
          dest_config = self.fetch_gcs_file(gcs_rel)
          if dest_config and dest_config.exists():
            try:
              raw_config = auto_model.AutoModel.load_config(
                  str(dest_config.parent)
              )
              resolved_subfolder = cand
              break
            except Exception as exc:  # pylint: disable=broad-except
              logging.warning(
                  "Failed to load GCS diffusers config from %s: %s",
                  dest_config.parent,
                  exc,
              )

      if raw_config is None:
        if load_weights:
          # Pretrained weights are not available in local resources or GCS.
          raise ValueError(
              f"load_weights cannot be set to True for {name} when model is"
              " not available in cache."
          )
        raise ValueError(
            f"Model config for '{name}' is missing in local resources"
            f" ({self._FILES.joinpath(str(pathlib.Path(name) / 'config.json'))})"
            " and could not be loaded from cache or GCS."
        )

      return _ConfigCacheEntry(
          config=raw_config,
          path=model_path,
          subfolder=resolved_subfolder,
      )

    entry = _CONFIG_CACHE.get(
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

    def _module_factory():
      if load_weights:
        return auto_model.AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=d_type,
            subfolder=subfolder,
        )
      else:
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
        model = model_cls.from_config(config_dict)
        if isinstance(model, tuple):
          model = model[0]
        return model.to(d_type)

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

      noisy_latents = torch.randn(
          (batch_size, latent_channels, *latent_dims),
          generator=g,
          dtype=d_type,
      ).to(device)

      # standard num_train_timesteps for diffusion models is typically 1000
      timesteps = torch.randint(
          0, 1000, (batch_size,), generator=g, dtype=torch.long
      ).to(device)

      # Dynamically get the model class to inspect its signature
      class_name = cfg.get("_class_name")
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
      forward_params = inspect.signature(model_cls.forward).parameters

      # Fallback to encoder_hid_dim / text_dim / joint_attention_dim if
      # cross_attention_dim is not specified in config.
      cross_attention_dim = (
          cfg.get("encoder_hid_dim")
          or cfg.get("cross_attention_dim")
          or cfg.get("joint_attention_dim")
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

      kwargs = {
          primary_input_key: noisy_latents,
          "timestep": timesteps,
          "encoder_hidden_states": dummy_encoder_hidden_states,
      }

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

      if "encoder_hidden_states" not in forward_params:
        kwargs.pop("encoder_hidden_states", None)
      if "added_cond_kwargs" not in forward_params:
        kwargs.pop("added_cond_kwargs", None)
      if "timestep" not in forward_params:
        kwargs.pop("timestep", None)
      if "class_labels" in forward_params or cfg.get("num_class_embeds"):
        kwargs["class_labels"] = torch.zeros(
            (batch_size,), dtype=torch.long, device=device
        )

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

  def list_all_sources(self) -> list[str]:
    """Returns a list of registered provider keys (e.g., 'torchvision')."""
    return list(self._providers.keys())

  def list_all_modules(self) -> list[str]:
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

  def list_modules(self, source: str) -> list[str]:
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
