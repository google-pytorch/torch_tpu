# Copyright 2025 Google LLC
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

"""Dummy input generation for Hugging Face Transformers models."""

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
import types
from typing import Any, overload

import torch

try:
  import transformers

  _HAS_TRANSFORMERS = True
except ImportError:
  _HAS_TRANSFORMERS = False

_MAX_SEQ_LEN_HEURISTIC_CAP = 1_000_000

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
    "qwen2_5_vl",
    "qwen2_audio",
    "qwen2_vl",
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

_VIDEO_MODEL_TYPES: frozenset[str] = frozenset({
    # go/keep-sorted start
    "timesformer",
    "videomae",
    "videomt",
    "videoprism",
    "vivit",
    "vjepa2",
    "xclip",
    # go/keep-sorted end
})

_STEREO_PAIR_MODEL_TYPES: frozenset[str] = frozenset({
    # go/keep-sorted start
    "efficientloftr",
    "lightglue",
    "superglue",
    # go/keep-sorted end
})

_DUAL_ENCODER_MODEL_TYPES: frozenset[str] = frozenset({
    # go/keep-sorted start
    "align",
    "altclip",
    "bridgetower",
    "chinese_clip",
    "clap",
    "clip",
    "flava",
    "groupvit",
    "siglip",
    "vision-text-dual-encoder",
    "vision_text_dual_encoder",
    # go/keep-sorted end
})

_PATCHIFIED_VLM_MODEL_TYPES: frozenset[str] = frozenset({
    # go/keep-sorted start
    "colqwen",
    "colqwen2",
    "glm4v",
    "glm_ocr",
    "holo",
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_5",
    "qwen3_vl",
    "qwen_vl",
    "qwenvl",
    # go/keep-sorted end
})

# Model types excluded from image_token_id injection at HEAD (expanded in CL 6).
_LEGACY_NO_INJECT_VLM_MODEL_TYPES: frozenset[str] = frozenset({
    # go/keep-sorted start
    "colqwen",
    "glm4v",
    "glm_ocr",
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen_vl",
    "qwenvl",
    # go/keep-sorted end
})

_TIME_SERIES_MODEL_TYPES: frozenset[str] = frozenset({
    # go/keep-sorted start
    "autoformer",
    "informer",
    "patchtsmixer",
    "patchtst",
    "time-series-transformer",
    "time_series_transformer",
    "timesfm",
    # go/keep-sorted end
})


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

  def __contains__(self, key: str) -> bool:
    data = self._data
    if isinstance(data, (dict, Mapping)):
      return key in data
    if hasattr(data, "_mock_name") and key not in getattr(data, "__dict__", {}):
      return False
    return hasattr(data, key)


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
  for sub in ("text_config", "encoder", "decoder", "thinker_config"):
    sub_cfg = cfg.get(sub)
    if sub_cfg is not None:
      wrapped_sub = _wrap_config(sub_cfg)
      configs_to_check.append(wrapped_sub)
      nested_text = wrapped_sub.get("text_config")
      if nested_text is not None:
        configs_to_check.append(_wrap_config(nested_text))

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

  model_type = cfg.get("model_type", "").lower()
  is_unified = "unified" in model_type
  if is_unified and str(device).startswith("meta"):
    # Gemma4UnifiedModel.get_image_features calls .tolist() on a boolean mask
    # tensor derived from image_position_ids and checks masked numel() equality,
    # both of which fail on meta tensors. Skip optional vision inputs on meta
    # device so the text backbone forward pass runs without monkey-patching.
    return

  # Default max_soft_tokens from Gemma4ImageProcessor or
  # Gemma4UnifiedImageProcessor.
  # This is the budget of soft tokens for the model.
  max_soft_tokens = 280

  num_channels = (
      cfg.get("num_channels")
      or (vision_config.get("num_channels") if vision_config else None)
      or 3
  )

  if is_unified:
    # Gemma 4 Unified models have no vision encoder; patches are pre-merged
    # into model patches of size (patch_size * pooling_kernel_size) with 6912
    # channels.
    model_patch_size = patch_size * pooling_kernel_size
    patch_pixels = num_channels * model_patch_size * model_patch_size
    max_patches = max_soft_tokens
    grid_size = image_size // model_patch_size
    num_real_patches = grid_size * grid_size
    if num_real_patches > max_patches:
      num_real_patches = max_patches
      grid_size = int(num_real_patches**0.5)
      num_real_patches = grid_size * grid_size
    num_features = num_real_patches
  else:
    max_patches = max_soft_tokens * pooling_kernel_size**2
    grid_size = image_size // patch_size
    num_real_patches = grid_size * grid_size
    if num_real_patches > max_patches:
      num_real_patches = max_patches
      grid_size = int(num_real_patches**0.5)
      num_real_patches = grid_size * grid_size
    patch_pixels = num_channels * patch_size * patch_size
    pooled_dim = grid_size // pooling_kernel_size
    num_features = pooled_dim * pooled_dim

  input_kwargs["pixel_values"] = torch.randn(
      batch_size, max_patches, patch_pixels, generator=generator
  ).to(device)

  grid_x, grid_y = torch.meshgrid(
      torch.arange(grid_size),
      torch.arange(grid_size),
      indexing="xy" if is_unified else "ij",
  )
  coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)
  coords_batch = coords.unsqueeze(0).expand(batch_size, -1, -1).to(device)
  pad_count = max_patches - num_real_patches
  if pad_count > 0:
    pad_tensor = torch.full(
        (batch_size, pad_count, 2), -1, device=device, dtype=torch.long
    )
    image_position_ids = torch.cat([coords_batch, pad_tensor], dim=1)
  else:
    image_position_ids = coords_batch

  image_token_id = cfg.get("image_token_id", 258880)
  image_tokens = torch.full(
      (batch_size, num_features),
      image_token_id,
      device=device,
      dtype=torch.long,
  )
  orig_ids = input_kwargs["input_ids"]
  input_kwargs["input_ids"] = torch.cat(
      [image_tokens, orig_ids[:, num_features:]], dim=-1
  )
  if "attention_mask" in input_kwargs:
    input_kwargs["attention_mask"] = torch.ones_like(input_kwargs["input_ids"])
  input_kwargs["image_position_ids"] = image_position_ids


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


@dataclasses.dataclass(frozen=True)
class InputContext:
  """Centralized per-call state and tensor factory for input generation.

  Provides centralized control of tensor creation across all modality input
  builders to ensure reproducible RNG state (deterministic CPU generator),
  dtype propagation, and device placement.
  """

  config: Any  # _ConfigWrapper
  model_type: str  # lower-cased
  arch_name: str  # lower-cased architectures[0] or ""
  type_str: str  # f"{model_type} {arch_name}" (existing convention)
  batch_size: int  # shape[0] if shape else 1
  shape: Sequence[int] | None
  device: str
  generator: torch.Generator  # CPU generator seeded 42 (existing convention)
  dtype: torch.dtype | None = None  # diffusers passes bfloat16

  @classmethod
  def create(
      cls,
      config: Any,
      shape: Sequence[int] | None,
      device: str,
      dtype: torch.dtype | None = None,
  ) -> "InputContext":
    config = _wrap_config(config)
    g = torch.Generator(device="cpu").manual_seed(42)
    raw_model_type = config.get("model_type", "unknown")
    if isinstance(raw_model_type, _ConfigWrapper):
      raw_model_type = raw_model_type.unwrap()
    model_type = (
        raw_model_type.lower()
        if isinstance(raw_model_type, str)
        else str(raw_model_type).lower()
    )
    raw_archs = config.get("architectures", [])
    if isinstance(raw_archs, _ConfigWrapper):
      raw_archs = raw_archs.unwrap()
    if isinstance(raw_archs, (list, tuple)) and raw_archs:
      raw_arch_name = raw_archs[0]
      if isinstance(raw_arch_name, _ConfigWrapper):
        raw_arch_name = raw_arch_name.unwrap()
      arch_name = (
          raw_arch_name.lower()
          if isinstance(raw_arch_name, str)
          else str(raw_arch_name).lower()
      )
    else:
      arch_name = ""
    type_str = f"{model_type} {arch_name}"
    batch_size = shape[0] if shape else 1
    return cls(
        config=config,
        model_type=model_type,
        arch_name=arch_name,
        type_str=type_str,
        batch_size=batch_size,
        shape=shape,
        device=device,
        generator=g,
        dtype=dtype,
    )

  def randn(
      self,
      *size: int,
      dtype: torch.dtype | None = None,
  ) -> torch.Tensor:
    t = torch.randn(*size, generator=self.generator, dtype=dtype)
    return t.to(self.device)

  def randint(
      self,
      low: int,
      high: int,
      size: Sequence[int] | tuple[int, ...],
      dtype: torch.dtype = torch.long,
  ) -> torch.Tensor:
    t = torch.randint(
        low, high, tuple(size), generator=self.generator, dtype=dtype
    )
    return t.to(self.device)

  def ones(
      self,
      size: Sequence[int] | tuple[int, ...],
      dtype: torch.dtype = torch.long,
  ) -> torch.Tensor:
    return torch.ones(tuple(size), device=self.device, dtype=dtype)

  def zeros(
      self,
      size: Sequence[int] | tuple[int, ...],
      dtype: torch.dtype = torch.long,
  ) -> torch.Tensor:
    return torch.zeros(tuple(size), device=self.device, dtype=dtype)

  def full(
      self,
      size: Sequence[int] | tuple[int, ...],
      fill_value: Any,
      dtype: torch.dtype = torch.long,
  ) -> torch.Tensor:
    return torch.full(tuple(size), fill_value, device=self.device, dtype=dtype)


def _cfg_lookup(cfg: Any, *paths: str, default: Any = None) -> Any:
  """First non-None value among dotted paths, e.g.

  ("vision_config.patch_size", "patch_size").
  """
  cfg = _wrap_config(cfg)
  for path in paths:
    cur = cfg
    for part in path.split("."):
      if cur is None:
        break
      if isinstance(cur, _ConfigWrapper):
        cur = cur.get(part)
      elif isinstance(cur, Mapping):
        cur = cur.get(part)
      elif hasattr(cur, "_mock_name") and part not in getattr(
          cur, "__dict__", {}
      ):
        cur = None
        break
      elif hasattr(cur, part):
        cur = getattr(cur, part)
      elif hasattr(cur, "get"):
        cur = cur.get(part)
      else:
        cur = None
        break
    if cur is not None:
      if isinstance(cur, _ConfigWrapper):
        return cur.unwrap()
      return cur
  return default


@overload
def _cfg_int(
    cfg: Any,
    *paths: str,
    default: int,
    min_val: int = 1,
) -> int:
  ...


@overload
def _cfg_int(
    cfg: Any,
    *paths: str,
    default: None = None,
    min_val: int = 1,
) -> int | None:
  ...


def _cfg_int(
    cfg: Any,
    *paths: str,
    default: int | None = None,
    min_val: int = 1,
) -> int | None:
  """_safe_int(_cfg_lookup(...)) helper for config access."""
  val = _cfg_lookup(cfg, *paths, default=None)
  if val is None:
    return default
  return _safe_int(val, default=default, min_val=min_val)


def _resolve_seq_len(
    ctx: InputContext,
    *,
    cap: int = 512,
    use_shape: bool = True,
) -> int:
  """min(shape[1], max_pos) when a shape is given, else min(max_pos, cap)."""
  max_pos = _get_max_seq_len(ctx.config)
  if "videoprism" in ctx.type_str:
    cap = min(cap, 8)
  if "prophetnet" in ctx.type_str:
    cap = min(cap, 16)
  if "encoder-decoder" in ctx.type_str or "gpt2-bert" in ctx.type_str:
    cap = min(cap, 32)
  if use_shape and ctx.shape is not None and len(ctx.shape) > 1:
    if "prophetnet" in ctx.type_str:
      return min(ctx.shape[1], max_pos, cap)
    return min(ctx.shape[1], max_pos)
  return min(max_pos, cap)


def _build_text_tokens(
    ctx: InputContext,
    *,
    seq_len: int,
    vocab_size: int,
    min_token_id: int = 0,
    eos_token_id: int | None = None,
) -> dict[str, torch.Tensor]:
  """Returns {"input_ids", "attention_mask"} text tokens."""
  if "multiplechoice" in ctx.type_str:
    actual_shape = (ctx.batch_size, 2, seq_len)
  else:
    actual_shape = (ctx.batch_size, seq_len)
  input_ids = ctx.randint(min_token_id, vocab_size, actual_shape)
  attention_mask = ctx.ones(actual_shape)
  if eos_token_id is not None and 0 <= eos_token_id < vocab_size:
    input_ids[..., -1] = eos_token_id
  return {"input_ids": input_ids, "attention_mask": attention_mask}


def _ensure_min_seq_len(
    ctx: InputContext,
    input_kwargs: dict[str, Any],
    *,
    needed_len: int,
    vocab_size: int,
    target_len: int | None = None,
) -> int:
  """Regenerates input_ids/attention_mask if shorter than needed."""
  cur_seq_len = input_kwargs["input_ids"].shape[1]
  if cur_seq_len < needed_len:
    target = needed_len if target_len is None else target_len
    max_allowed = _get_max_seq_len(ctx.config)
    new_seq_len = min(target, max_allowed)
    input_kwargs["input_ids"] = ctx.randint(
        0, vocab_size, (ctx.batch_size, new_seq_len)
    )
    input_kwargs["attention_mask"] = ctx.ones((ctx.batch_size, new_seq_len))
    return new_seq_len
  return cur_seq_len


def _inject_token_span(
    input_ids: torch.Tensor,
    token_id: int,
    *,
    start: int,
    count: int,
) -> int:
  """Injects a span of token_id into input_ids; returns actual count."""
  cur_len = input_ids.shape[1]
  valid_count = min(count, max(0, cur_len - start))
  if valid_count > 0:
    replacement_id = 1 if token_id == 0 else 0
    input_ids.masked_fill_(input_ids == token_id, replacement_id)
    input_ids[:, start : start + valid_count] = token_id
  return valid_count


def _mm_token_type_ids(
    input_ids: torch.Tensor,
    *,
    start: int,
    count: int,
) -> torch.Tensor:
  """Builds mm_token_type_ids tensor with 1s in [start, start + count)."""
  mm_types = torch.zeros_like(input_ids, dtype=torch.int)
  valid_count = min(count, max(0, input_ids.shape[1] - start))
  if valid_count > 0:
    mm_types[:, start : start + valid_count] = 1
  return mm_types


def _build_pixel_values(
    ctx: InputContext,
    *,
    num_channels: int,
    image_size: int,
    num_images: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
  h = height if height is not None else image_size
  w = width if width is not None else image_size
  if num_frames is not None:
    return ctx.randn(ctx.batch_size, num_frames, num_channels, h, w)
  if num_images is not None:
    return ctx.randn(ctx.batch_size, num_images, num_channels, h, w)
  return ctx.randn(ctx.batch_size, num_channels, h, w)


def _build_siglip2_inputs(
    ctx: InputContext,
    input_kwargs: dict[str, Any],
    *,
    image_size: int | None = None,
) -> None:
  img_size = (
      image_size if image_size is not None else _parse_image_size(ctx.config)
  )
  patch_size = _cfg_int(
      ctx.config, "vision_config.patch_size", "patch_size", default=16
  )
  num_channels = _get_num_channels(ctx.config)
  grid_size = img_size // patch_size
  num_patches = grid_size * grid_size
  patch_dim = num_channels * patch_size * patch_size
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size, num_patches, patch_dim
  )
  input_kwargs["pixel_attention_mask"] = ctx.ones(
      (ctx.batch_size, num_patches), dtype=torch.bool
  )
  # Siglip2VisionEmbeddings.resize_positional_embeddings calls
  # `height, width = spatial_shapes[i].tolist()` to pass integer dimensions to
  # F.interpolate, so keep spatial_shapes on CPU when evaluating on meta device.
  shapes_device = "cpu" if str(ctx.device).startswith("meta") else ctx.device
  input_kwargs["spatial_shapes"] = torch.tensor(
      [[grid_size, grid_size]] * ctx.batch_size,
      device=shapes_device,
      dtype=torch.long,
  )


def _build_oneformer_inputs(
    ctx: InputContext,
    input_kwargs: dict[str, Any],
) -> None:
  task_len = _safe_int(ctx.config.get("task_seq_len"), default=77, min_val=1)
  input_kwargs["task_inputs"] = torch.zeros(
      (ctx.batch_size, task_len), device=ctx.device, dtype=torch.long
  )


def _build_spatial_merged_patches(
    ctx: InputContext,
    input_kwargs: dict[str, Any],
    *,
    image_size: int,
    num_channels: int,
) -> int:
  """Flattened patches + image_grid_thw; returns num_img_tokens."""
  vision_config = ctx.config.get("vision_config")
  patch_size = _cfg_int(vision_config, "patch_size", default=14)
  temporal_patch_size = _cfg_int(
      vision_config, "temporal_patch_size", default=2
  )
  spatial_merge_size = _cfg_int(vision_config, "spatial_merge_size", default=2)
  patches_per_side = image_size // patch_size
  num_img_tokens = (patches_per_side // spatial_merge_size) ** 2

  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size * patches_per_side * patches_per_side,
      num_channels * patch_size * patch_size * temporal_patch_size,
  )
  input_kwargs["image_grid_thw"] = torch.tensor(
      [[1, patches_per_side, patches_per_side]] * ctx.batch_size,
      device=ctx.device,
      dtype=torch.long,
  )
  return num_img_tokens


def _add_encoder_decoder_inputs(
    ctx: InputContext,
    input_kwargs: dict[str, Any],
    modality: Any,
) -> None:
  """Adds decoder_input_ids for encoder-decoder models."""
  mod_key = (
      modality.value if hasattr(modality, "value") else str(modality).lower()
  )
  if "." in mod_key:
    mod_key = mod_key.split(".")[-1]
  if (
      bool(ctx.config.get("is_encoder_decoder", False))
      and "decoder_input_ids" not in input_kwargs
      and mod_key != "vision"
      and ctx.model_type not in ("speecht5", "vits", "musicgen")
      and "musicgen" not in ctx.type_str
  ):
    text_cfg = ctx.config.get("text_config")
    gen_cfg = ctx.config.get("generator") or ctx.config.get("generator_config")
    decoder_cfg = ctx.config.get("decoder") or ctx.config.get("decoder_config")
    vocab_size = _safe_int(
        gen_cfg.get("vocab_size")
        if "rag" in ctx.type_str and gen_cfg
        else (
            ctx.config.get("tgt_vocab_size")
            or (decoder_cfg.get("vocab_size") if decoder_cfg else None)
            or ctx.config.get("vocab_size")
            or (text_cfg.get("vocab_size") if text_cfg else None)
        ),
        default=32000,
        min_val=1,
    )

    max_len = _get_max_seq_len(ctx.config)
    if "whisper" in ctx.model_type:
      max_len = min(max_len, 448)
    max_target_pos = _safe_int(
        ctx.config.get("max_target_positions"),
        default=None,
        min_val=1,
    )
    if max_target_pos is not None:
      max_len = min(max_len, max_target_pos)
    seq_limit = ctx.shape[1] if ctx.shape and len(ctx.shape) > 1 else 512
    safe_seq_len = min(max_len, seq_limit)
    if "prophetnet" in ctx.type_str:
      safe_seq_len = min(safe_seq_len, 16)
    if "encoder-decoder" in ctx.type_str or "gpt2-bert" in ctx.type_str:
      safe_seq_len = min(safe_seq_len, 32)
    # Use first dimension of shape or 1 for batch size
    batch_size = ctx.batch_size
    decoder_shape = (batch_size, safe_seq_len)
    input_kwargs["decoder_input_ids"] = ctx.randint(
        0,
        vocab_size,
        decoder_shape,
    )


def _cast_floating_point_inputs(
    config: Any,
    input_kwargs: dict[str, Any],
) -> dict[str, Any]:
  """Casts float tensors in input_kwargs to model target dtype."""
  target_dtype = _extract_target_dtype(config)

  def _cast_floating_point(val: Any) -> Any:
    if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
      return val.to(dtype=target_dtype)
    if isinstance(val, dict):
      return {k: _cast_floating_point(sub_v) for k, sub_v in val.items()}
    if isinstance(val, tuple):
      return tuple(_cast_floating_point(item) for item in val)
    if isinstance(val, list):
      return [_cast_floating_point(item) for item in val]
    return val

  for k, v in list(input_kwargs.items()):
    input_kwargs[k] = _cast_floating_point(v)

  return input_kwargs


_Matcher = Callable[[InputContext], bool]
_Builder = Callable[[InputContext, dict[str, Any]], None]


def _type_contains(*needles: str) -> _Matcher:
  return lambda ctx: any(n in ctx.type_str for n in needles)


def _model_type_contains(*needles: str) -> _Matcher:
  return lambda ctx: any(n in ctx.model_type for n in needles)


def _has_config_key(*paths: str) -> _Matcher:
  return lambda ctx: any(_cfg_lookup(ctx.config, p) is not None for p in paths)


def _apply_builders(
    ctx: InputContext,
    input_kwargs: dict[str, Any],
    table: Sequence[tuple[_Matcher, _Builder]],
    *,
    first_match_only: bool,
) -> None:
  """`elif` ladders become first_match_only=True tables; independent `if`s

  become first_match_only=False tables. Row order == original statement order
  (behavior preserving).
  """
  for matcher, builder in table:
    if matcher(ctx):
      builder(ctx, input_kwargs)
      if first_match_only:
        break


# ---------------------------------------------------------------------------
# Text Modality Builders
# ---------------------------------------------------------------------------


def _build_timesfm_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  context_length = _cfg_int(
      ctx.config, "context_length", "seq_length", default=64
  )
  input_kwargs.pop("input_ids", None)
  input_kwargs.pop("attention_mask", None)
  input_kwargs["past_values"] = ctx.randn(ctx.batch_size, context_length)
  input_kwargs["freq"] = ctx.zeros((ctx.batch_size,))


def _build_time_series_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  context_length = _cfg_int(
      ctx.config, "context_length", "seq_length", default=64
  )
  input_size = _cfg_int(
      ctx.config, "input_size", "num_input_channels", default=1
  )
  num_time_features = _cfg_int(ctx.config, "num_time_features", default=4)
  input_kwargs.pop("input_ids", None)
  input_kwargs.pop("attention_mask", None)
  lags_sequence = ctx.config.get("lags_sequence", [1, 2, 3, 4, 5, 6, 7])
  max_lag = max(lags_sequence) if lags_sequence else 0
  if any(
      k in ctx.type_str
      for k in (
          "autoformer",
          "informer",
          "time_series_transformer",
          "time-series-transformer",
      )
  ):
    past_length = context_length + max_lag
    num_feat = num_time_features + _cfg_int(
        ctx.config, "num_dynamic_real_features", default=0, min_val=0
    )
    val_shape = (
        (ctx.batch_size, past_length)
        if input_size == 1
        else (ctx.batch_size, past_length, input_size)
    )
    input_kwargs["past_values"] = ctx.randn(*val_shape)
    input_kwargs["past_time_features"] = ctx.randn(
        ctx.batch_size, past_length, num_feat
    )
    input_kwargs["past_observed_mask"] = torch.ones(
        val_shape,
        device=ctx.device,
        dtype=torch.bool,
    )
    num_static_cat = _cfg_int(
        ctx.config, "num_static_categorical_features", default=0, min_val=0
    )
    if num_static_cat > 0:
      input_kwargs["static_categorical_features"] = ctx.zeros(
          (ctx.batch_size, num_static_cat), dtype=torch.long
      )
    num_static_real = _cfg_int(
        ctx.config, "num_static_real_features", default=0, min_val=0
    )
    if num_static_real > 0:
      input_kwargs["static_real_features"] = ctx.zeros(
          (ctx.batch_size, num_static_real), dtype=torch.float32
      )
    return

  context_length = max(context_length, max_lag + 1, 64)
  input_kwargs["past_values"] = ctx.randn(
      ctx.batch_size, context_length, input_size
  )
  input_kwargs["past_time_features"] = ctx.randn(
      ctx.batch_size, context_length, num_time_features
  )
  input_kwargs["past_observed_mask"] = torch.ones(
      (ctx.batch_size, context_length, input_size),
      device=ctx.device,
      dtype=torch.bool,
  )


def _build_decision_transformer_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs.pop("input_ids", None)
  state_dim = _cfg_int(ctx.config, "state_dim", default=17)
  act_dim = _cfg_int(ctx.config, "act_dim", default=4)
  seq_len = _resolve_seq_len(ctx, cap=512)
  input_kwargs["states"] = ctx.randn(ctx.batch_size, seq_len, state_dim)
  input_kwargs["actions"] = ctx.randn(ctx.batch_size, seq_len, act_dim)
  input_kwargs["returns_to_go"] = ctx.randn(ctx.batch_size, seq_len, 1)
  input_kwargs["timesteps"] = torch.arange(
      seq_len, device=ctx.device, dtype=torch.long
  ).repeat(ctx.batch_size, 1)
  input_kwargs["attention_mask"] = ctx.ones((ctx.batch_size, seq_len))


def _build_luke_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_entities = 2 if "entitypair" in ctx.arch_name else 1
  input_kwargs["entity_ids"] = ctx.zeros((ctx.batch_size, num_entities))
  input_kwargs["entity_attention_mask"] = ctx.ones(
      (ctx.batch_size, num_entities)
  )
  input_kwargs["entity_position_ids"] = ctx.zeros(
      (ctx.batch_size, num_entities, 14)
  )


def _build_bros_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  seq_len = input_kwargs["input_ids"].shape[1]
  input_kwargs["bbox"] = ctx.zeros((ctx.batch_size, seq_len, 4))


def _build_lxmert_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  feat_dim = _cfg_int(ctx.config, "visual_feat_dim", default=2048)
  input_kwargs["visual_feats"] = ctx.randn(ctx.batch_size, 36, feat_dim)
  input_kwargs["visual_pos"] = ctx.zeros(
      (ctx.batch_size, 36, 4), dtype=torch.float32
  )


def _build_tapas_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  seq_len = input_kwargs["input_ids"].shape[1]
  token_type_ids_tensor = ctx.zeros((ctx.batch_size, seq_len, 7))
  token_type_ids_tensor[:, :, 0] = 1  # segment
  col_indices = torch.arange(seq_len, device=ctx.device) % 2
  row_indices = (torch.arange(seq_len, device=ctx.device) // 2) % 2
  token_type_ids_tensor[:, :, 1] = col_indices
  token_type_ids_tensor[:, :, 2] = row_indices
  input_kwargs["token_type_ids"] = token_type_ids_tensor


def _build_mobilebert_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  seq_len = input_kwargs["input_ids"].shape[1]
  input_kwargs["token_type_ids"] = ctx.zeros((ctx.batch_size, seq_len))


def _build_rag_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  n_docs = _cfg_int(ctx.config, "n_docs", default=5)
  q_vocab_size = _cfg_int(
      ctx.config,
      "question_encoder.vocab_size",
      "question_encoder_config.vocab_size",
      default=30522,
  )
  seq_len = input_kwargs["input_ids"].shape[1]
  input_kwargs["input_ids"] = ctx.randint(
      0, q_vocab_size, (ctx.batch_size, seq_len)
  )
  gen_vocab_size = _cfg_int(
      ctx.config,
      "generator.vocab_size",
      "generator_config.vocab_size",
      default=50265,
  )
  input_kwargs["context_input_ids"] = ctx.randint(
      0,
      gen_vocab_size,
      (ctx.batch_size * n_docs, seq_len),
  )
  input_kwargs["context_attention_mask"] = ctx.ones(
      (ctx.batch_size * n_docs, seq_len)
  )
  input_kwargs["doc_scores"] = ctx.randn(ctx.batch_size, n_docs)


def _build_perceiver_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs.pop("input_ids", None)
  input_kwargs.pop("attention_mask", None)
  d_model = _cfg_int(ctx.config, "d_model", default=768)
  seq_len = _resolve_seq_len(ctx, cap=512)
  input_kwargs["inputs"] = ctx.randn(ctx.batch_size, seq_len, d_model)


def _build_gemma4_assistant_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  backbone_hidden_size = _cfg_int(
      ctx.config,
      "backbone_hidden_size",
      default=3840 if "unified" in ctx.model_type else 1536,
  )
  embed_dim = 2 * backbone_hidden_size
  seq_len = input_kwargs["input_ids"].shape[1]
  input_kwargs.pop("input_ids", None)
  input_kwargs["inputs_embeds"] = ctx.randn(ctx.batch_size, seq_len, embed_dim)
  text_cfg = ctx.config.get("text_config") or ctx.config
  layer_types = text_cfg.get(
      "layer_types", ["sliding_attention", "full_attention"]
  )
  all_layer_types = set(layer_types) | {
      "sliding_attention",
      "full_attention",
  }
  shared_kv_states = {}
  for layer_key in sorted(all_layer_types):
    layer_cfg = None
    per_layer_cfg = text_cfg.get("per_layer_config")
    if per_layer_cfg is not None:
      try:
        layer_cfg = per_layer_cfg[layer_key]
      except Exception:  # pylint: disable=broad-except
        layer_cfg = None
    if layer_cfg is None:
      layer_cfg = text_cfg

    is_global = "full" in layer_key or "global" in layer_key
    if is_global:
      h_dim = _safe_int(
          layer_cfg.get("global_head_dim")
          or layer_cfg.get("head_dim")
          or (
              layer_cfg.get("hidden_size", 1024)
              // layer_cfg.get("num_attention_heads", 16)
          ),
          default=512,
          min_val=1,
      )
      n_kv = _safe_int(
          layer_cfg.get("num_global_key_value_heads")
          or layer_cfg.get("num_key_value_heads")
          or layer_cfg.get("num_attention_heads"),
          default=8,
          min_val=1,
      )
    else:
      h_dim = _safe_int(
          layer_cfg.get("head_dim")
          or (
              layer_cfg.get("hidden_size", 1024)
              // layer_cfg.get("num_attention_heads", 16)
          ),
          default=256,
          min_val=1,
      )
      n_kv = _safe_int(
          layer_cfg.get("num_key_value_heads")
          or layer_cfg.get("num_attention_heads"),
          default=8,
          min_val=1,
      )
    k = ctx.randn(ctx.batch_size, n_kv, seq_len, h_dim)
    v = ctx.randn(ctx.batch_size, n_kv, seq_len, h_dim)
    shared_kv_states[layer_key] = (k, v)
  input_kwargs["shared_kv_states"] = shared_kv_states


def _build_xlstm_text_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  chunk_size = _cfg_int(ctx.config, "chunk_size", default=64)
  vocab_size = _cfg_int(ctx.config, "vocab_size", default=50304)
  seq_len = max(
      chunk_size,
      (input_kwargs["input_ids"].shape[1] // chunk_size) * chunk_size,
  )
  input_kwargs["input_ids"] = ctx.randint(
      0, vocab_size, (ctx.batch_size, seq_len)
  )
  input_kwargs.pop("attention_mask", None)


_TEXT_EXTRA_BUILDERS: tuple[tuple[_Matcher, _Builder], ...] = (
    (_type_contains("timesfm"), _build_timesfm_text_inputs),
    (
        lambda ctx: any(k in ctx.type_str for k in _TIME_SERIES_MODEL_TYPES),
        _build_time_series_text_inputs,
    ),
    (
        _type_contains("decision_transformer"),
        _build_decision_transformer_text_inputs,
    ),
    (_type_contains("luke"), _build_luke_text_inputs),
    (_type_contains("bros"), _build_bros_text_inputs),
    (_type_contains("lxmert"), _build_lxmert_text_inputs),
    (_model_type_contains("tapas"), _build_tapas_text_inputs),
    (_model_type_contains("mobilebert"), _build_mobilebert_text_inputs),
    (_type_contains("rag"), _build_rag_text_inputs),
    (
        lambda ctx: (
            "perceiver" in ctx.type_str and ctx.arch_name == "perceivermodel"
        ),
        _build_perceiver_text_inputs,
    ),
    (
        _type_contains("gemma4_assistant", "gemma4_unified_assistant"),
        _build_gemma4_assistant_text_inputs,
    ),
    (_model_type_contains("xlstm"), _build_xlstm_text_inputs),
)


def _build_text_inputs(ctx: InputContext) -> dict[str, Any]:
  """Builds inputs for text / causal_lm / seq2seq models."""
  safe_seq_len = _resolve_seq_len(ctx, cap=512)
  # Prefer `src_vocab_size` over `vocab_size` for asymmetric encoder-decoder
  # models such as FSMT (`facebook/wmt19-ru-en`), where `config.vocab_size`
  # aliases `tgt_vocab_size` (31640) via `FSMTConfig.attribute_map` while
  # `FSMTEncoder.embed_tokens` only has `src_vocab_size` (31232) rows.
  vocab_size = _cfg_int(
      ctx.config,
      "src_vocab_size",
      "encoder.vocab_size",
      "encoder_config.vocab_size",
      "vocab_size",
      "text_config.vocab_size",
      default=32000,
  )
  eos_token_id = ctx.config.get("eos_token_id", 2)
  if isinstance(eos_token_id, (list, tuple)):
    eos_token_id = eos_token_id[0] if eos_token_id else 2
  eos_token_id = _safe_int(eos_token_id, default=2)

  input_kwargs = _build_text_tokens(
      ctx,
      seq_len=safe_seq_len,
      vocab_size=vocab_size,
      eos_token_id=eos_token_id,
  )
  _apply_builders(
      ctx, input_kwargs, _TEXT_EXTRA_BUILDERS, first_match_only=True
  )
  if "flaubert" in ctx.type_str:
    input_kwargs["use_cache"] = False
  if "ibert" in ctx.type_str and "input_ids" in input_kwargs:
    seq_len = input_kwargs["input_ids"].shape[1]
    input_kwargs["position_ids"] = (
        torch.arange(seq_len, device=ctx.device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(input_kwargs["input_ids"].shape[0], 1)
    )
  return input_kwargs


# ---------------------------------------------------------------------------
# Vision Modality Builders
# ---------------------------------------------------------------------------


def _build_seggpt_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_channels = _get_num_channels(ctx.config)
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size, num_channels, 448, 448
  )
  input_kwargs["prompt_pixel_values"] = ctx.randn(
      ctx.batch_size, num_channels, 448, 448
  )
  input_kwargs["prompt_masks"] = ctx.zeros(
      (ctx.batch_size, num_channels, 448, 448), dtype=torch.float32
  )


def _build_qwen3_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  patch_size = _cfg_int(ctx.config, "patch_size", default=14)
  temporal_patch_size = _cfg_int(ctx.config, "temporal_patch_size", default=2)
  grid_h = image_size // patch_size
  grid_w = image_size // patch_size
  num_patches = grid_h * grid_w
  in_channels = _cfg_int(ctx.config, "in_channels", default=num_channels)
  input_kwargs["hidden_states"] = ctx.randn(
      ctx.batch_size * num_patches,
      in_channels * temporal_patch_size * patch_size * patch_size,
  )
  input_kwargs["grid_thw"] = torch.tensor(
      [[1, grid_h, grid_w]] * ctx.batch_size,
      device=ctx.device,
      dtype=torch.long,
  )
  input_kwargs.pop("pixel_values", None)


def _build_mllama_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  num_images = 1
  num_tiles = _cfg_int(ctx.config, "max_num_tiles", default=4)
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size,
      num_images,
      num_tiles,
      num_channels,
      image_size,
      image_size,
  )
  input_kwargs["aspect_ratio_ids"] = ctx.ones((ctx.batch_size, num_images))
  input_kwargs["aspect_ratio_mask"] = ctx.ones(
      (ctx.batch_size, num_images, num_tiles)
  )


def _build_videomae_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  if "videoclassification" not in ctx.type_str:
    image_size = _parse_image_size(ctx.config)
    patch_size = _cfg_int(ctx.config, "patch_size", default=16)
    tubelet_size = _cfg_int(ctx.config, "tubelet_size", default=2)
    num_frames = _cfg_int(ctx.config, "num_frames", default=8)
    num_patches = ((image_size // patch_size) ** 2) * (
        num_frames // tubelet_size
    )
    num_masked = max(1, num_patches // 2)
    mask = torch.zeros(
        (ctx.batch_size, num_patches), device=ctx.device, dtype=torch.bool
    )
    mask[:, :num_masked] = True
    input_kwargs["bool_masked_pos"] = mask


def _build_vitpose_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["dataset_index"] = torch.tensor(
      [0] * ctx.batch_size, device=ctx.device, dtype=torch.long
  )


def _build_gemma4_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  patch_size = _cfg_int(ctx.config, "patch_size", default=16)
  pooling_kernel_size = _cfg_int(ctx.config, "pooling_kernel_size", default=3)
  num_channels = _get_num_channels(ctx.config)
  raw_grid = max(image_size // patch_size, pooling_kernel_size)
  grid_size = (
      (raw_grid + pooling_kernel_size - 1) // pooling_kernel_size
  ) * pooling_kernel_size
  num_patches = grid_size * grid_size
  patch_dim = num_channels * patch_size * patch_size
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size, num_patches, patch_dim
  )
  grid_xy = torch.stack(
      torch.meshgrid(
          torch.arange(grid_size, device=ctx.device, dtype=torch.long),
          torch.arange(grid_size, device=ctx.device, dtype=torch.long),
          indexing="ij",
      ),
      dim=-1,
  ).reshape(1, num_patches, 2)
  input_kwargs["pixel_position_ids"] = grid_xy.expand(
      ctx.batch_size, -1, -1
  ).contiguous()


def _build_safety_checker_vision_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  input_kwargs["clip_input"] = ctx.randn(
      ctx.batch_size, num_channels, image_size, image_size
  )
  input_kwargs["images"] = ctx.zeros(
      (ctx.batch_size, image_size, image_size, num_channels),
      dtype=torch.float32,
  )


def _build_vision_encoder_decoder_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  decoder_config = ctx.config.get("decoder")
  vocab_size = _cfg_int(decoder_config, "vocab_size", default=50265)
  input_kwargs["decoder_input_ids"] = ctx.randint(
      0, vocab_size, (ctx.batch_size, 8)
  )


_VISION_EXTRA_BUILDERS: tuple[tuple[_Matcher, _Builder], ...] = (
    (_type_contains("seggpt"), _build_seggpt_vision_inputs),
    (_type_contains("mllama"), _build_mllama_vision_inputs),
    (
        lambda ctx: "qwen3" in ctx.type_str and "vision" in ctx.type_str,
        _build_qwen3_vision_inputs,
    ),
    (_type_contains("videomae"), _build_videomae_vision_inputs),
    (_type_contains("vitpose"), _build_vitpose_vision_inputs),
    (
        _type_contains("siglip2"),
        lambda ctx, kw: _build_siglip2_inputs(ctx, kw),
    ),
    (_type_contains("oneformer"), _build_oneformer_inputs),
    (
        lambda ctx: "gemma4" in ctx.type_str and "vision" in ctx.type_str,
        _build_gemma4_vision_inputs,
    ),
    (
        _type_contains("safety_checker", "safety-checker"),
        _build_safety_checker_vision_inputs,
    ),
    (
        _model_type_contains(
            "trocr",
            "vision-encoder-decoder",
            "vision_encoder_decoder",
            "visionencoderdecoder",
        ),
        _build_vision_encoder_decoder_inputs,
    ),
)


def _build_vision_inputs(
    ctx: InputContext,
    model_dir: str | None = None,
    preprocessor_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
  """Builds inputs for vision models."""
  image_size = _parse_image_size(ctx.config)

  processor = None
  if model_dir and _HAS_TRANSFORMERS:
    if preprocessor_loader is not None:
      processor = preprocessor_loader(model_dir)
    else:
      try:
        processor = transformers.AutoProcessor.from_pretrained(model_dir)
      except Exception:  # pylint: disable=broad-except
        pass

  if (
      processor
      and hasattr(processor, "size")
      and isinstance(processor.size, dict)
  ):
    if "height" in processor.size:
      image_size = processor.size["height"]
    elif "shortest_edge" in processor.size:
      image_size = processor.size["shortest_edge"]

  num_channels = _get_num_channels(
      ctx.config, default_channels=4 if "vitmatte" in ctx.type_str else 3
  )

  raw_img_size = ctx.config.get("image_size")
  h, w = image_size, image_size
  if isinstance(raw_img_size, (list, tuple)) and len(raw_img_size) == 2:
    parsed_h = _safe_int(raw_img_size[0], default=None)
    parsed_w = _safe_int(raw_img_size[1], default=None)
    if parsed_h is not None and parsed_w is not None:
      h, w = parsed_h, parsed_w
  elif "vitpose" in ctx.type_str:
    h, w = 256, 192
  elif (
      "hear" in ctx.type_str
      or "hear"
      in str(
          ctx.config.get("name_or_path", "")
          or ctx.config.get("_name_or_path", "")
      ).lower()
  ):
    h, w = 192, 128

  input_kwargs: dict[str, Any] = {}
  if any(k in ctx.type_str for k in _VIDEO_MODEL_TYPES):
    num_frames = _cfg_int(
        ctx.config, "vision_config.num_frames", "num_frames", default=8
    )
    dummy_video = ctx.randn(ctx.batch_size, num_frames, num_channels, h, w)
    input_kwargs["pixel_values"] = dummy_video
    if "vjepa2" in ctx.type_str:
      input_kwargs["pixel_values_videos"] = dummy_video
    elif "videomt" in ctx.type_str or "videoprism" in ctx.type_str:
      input_kwargs["pixel_values_videos"] = dummy_video
      input_kwargs.pop("pixel_values", None)
  elif any(k in ctx.type_str for k in _STEREO_PAIR_MODEL_TYPES):
    input_kwargs["pixel_values"] = ctx.randn(
        ctx.batch_size, 2, num_channels, h, w
    )
  elif any(k in ctx.type_str for k in ["mgp_str", "mgp-str", "mgpstr"]):
    input_kwargs["pixel_values"] = ctx.randn(
        ctx.batch_size, num_channels, 32, 128
    )
  elif "pix2struct" in ctx.type_str:
    seq_len = 32
    patch_embed_hidden_size = _cfg_int(
        ctx.config, "patch_embed_hidden_size", default=768
    )
    input_kwargs["flattened_patches"] = ctx.randn(
        ctx.batch_size, seq_len, patch_embed_hidden_size + 2
    )
    input_kwargs["attention_mask"] = ctx.ones(
        (ctx.batch_size, seq_len), dtype=torch.long
    )
  else:
    input_kwargs["pixel_values"] = ctx.randn(ctx.batch_size, num_channels, h, w)

  _apply_builders(
      ctx, input_kwargs, _VISION_EXTRA_BUILDERS, first_match_only=False
  )
  return input_kwargs


# ---------------------------------------------------------------------------
# Audio Modality Builders
# ---------------------------------------------------------------------------


def _build_whisper_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_mel = _cfg_int(ctx.config, "num_mel_bins", default=80)
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, num_mel, 3000)


def _build_clap_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  audio_cfg = ctx.config.get("audio_config") or ctx.config
  num_mel = _safe_int(
      audio_cfg.get("num_mel_bins") or ctx.config.get("num_mel_bins"),
      default=64,
      min_val=1,
  )
  spec_size = _safe_int(
      audio_cfg.get("spec_size") or ctx.config.get("spec_size"),
      default=256,
      min_val=1,
  )
  freq_ratio = max(1, spec_size // num_mel)
  max_time = int(spec_size * freq_ratio)
  time_length = min(1001, max_time)
  input_kwargs["input_features"] = ctx.randn(
      ctx.batch_size, 1, time_length, num_mel
  )
  input_kwargs["is_longer"] = ctx.zeros((ctx.batch_size, 1), dtype=torch.bool)
  safe_seq_len = min(_get_max_seq_len(ctx.config), 512)
  vocab_size = _cfg_int(ctx.config, "vocab_size", default=32000)
  input_kwargs.update(
      _build_text_tokens(ctx, seq_len=safe_seq_len, vocab_size=vocab_size)
  )
  if any(
      k in ctx.arch_name
      for k in [
          "audio_encoder",
          "audioencoder",
          "audiomodel",
          "audio_model",
          "clapaudio",
          "clap_audio",
      ]
  ):
    input_kwargs.pop("input_ids", None)
    input_kwargs.pop("attention_mask", None)


def _build_ast_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  max_length = _cfg_int(ctx.config, "max_length", default=1024)
  num_mel = _cfg_int(ctx.config, "num_mel_bins", default=128)
  input_kwargs["input_values"] = ctx.randn(ctx.batch_size, max_length, num_mel)


def _build_w2v_bert_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  if any(
      k in ctx.type_str
      for k in [
          "qwen2_audio_encoder",
          "qwen2_audio",
          "qwen2-audio",
          "qwen2audio",
          "qwen2a",
      ]
  ):
    enc_cfg = ctx.config.get("encoder_config")
    num_mel = _safe_int(
        ctx.config.get("num_mel_bins")
        or (enc_cfg.get("num_mel_bins") if enc_cfg else None),
        default=128,
        min_val=1,
    )
    max_positions = _safe_int(
        ctx.config.get("max_source_positions"),
        default=1500,
        min_val=1,
    )
    input_kwargs["input_features"] = ctx.randn(
        ctx.batch_size, num_mel, max_positions * 2
    )
    return

  enc_cfg = (
      ctx.config.get("encoder_config")
      or ctx.config.get("encoder")
      or ctx.config
  )
  subsampling_channels = _cfg_lookup(
      ctx.config,
      "subsampling_conv_channels",
      "encoder_config.subsampling_conv_channels",
  )
  subsampling_in_mel = (
      subsampling_channels[0]
      if isinstance(subsampling_channels, (list, tuple))
      and subsampling_channels
      else None
  )
  num_mel = _safe_int(
      ctx.config.get("feature_projection_input_dim")
      or (enc_cfg.get("num_mel_bins") if enc_cfg else None)
      or (enc_cfg.get("input_feat_per_channel") if enc_cfg else None)
      or (enc_cfg.get("feature_size") if enc_cfg else None)
      or subsampling_in_mel,
      default=80,
      min_val=1,
  )
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, 128, num_mel)
  if any(k in ctx.type_str for k in ("parakeet", "nemotron")):
    input_kwargs["decoder_input_ids"] = ctx.zeros(
        (ctx.batch_size, 1), dtype=torch.long
    )
  if "nemotron" in ctx.type_str:
    input_kwargs["attention_mask"] = ctx.ones(
        (ctx.batch_size, 128), dtype=torch.long
    )
    input_kwargs["input_lengths"] = torch.tensor(
        [128] * ctx.batch_size, device=ctx.device, dtype=torch.long
    )


def _build_gemma3n_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["audio_mel"] = ctx.randn(ctx.batch_size, 128, 128)
  input_kwargs["audio_mel_mask"] = ctx.ones(
      (ctx.batch_size, 128), dtype=torch.bool
  )


def _build_omnitoken2wav_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["mel_spectrogram"] = ctx.randn(ctx.batch_size, 80, 128)


def _build_kyutai_stt_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_codebooks = _safe_int(
      ctx.config.get("num_codebooks"), default=32, min_val=1
  )
  seq_len = 16
  input_kwargs["input_ids"] = ctx.randint(
      0, 100, (ctx.batch_size, seq_len, num_codebooks + 1), dtype=torch.long
  )
  input_kwargs["attention_mask"] = ctx.ones(
      (ctx.batch_size, seq_len), dtype=torch.long
  )


def _build_moshi_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_codebooks = _safe_int(
      ctx.config.get("num_codebooks"), default=8, min_val=1
  )
  vocab_size = _safe_int(ctx.config.get("vocab_size"), default=32000, min_val=1)
  audio_vocab_size = _safe_int(
      ctx.config.get("audio_vocab_size"), default=2048, min_val=1
  )
  seq_len = 1
  input_kwargs["input_ids"] = ctx.randint(
      0, vocab_size, (ctx.batch_size, seq_len), dtype=torch.long
  )
  input_kwargs["user_audio_codes"] = ctx.randint(
      0,
      audio_vocab_size,
      (ctx.batch_size, num_codebooks, seq_len),
      dtype=torch.long,
  )
  input_kwargs["moshi_audio_codes"] = ctx.randint(
      0,
      audio_vocab_size,
      (ctx.batch_size, num_codebooks, seq_len),
      dtype=torch.long,
  )
  input_kwargs["attention_mask"] = ctx.ones(
      (ctx.batch_size, seq_len), dtype=torch.long
  )


def _build_glmasr_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  text_cfg = ctx.config.get("text_config") or ctx.config
  vocab_size = _safe_int(
      text_cfg.get("vocab_size") if text_cfg else None,
      default=151552,
      min_val=1,
  )
  input_kwargs["input_ids"] = ctx.randint(
      0, vocab_size, (ctx.batch_size, 8), dtype=torch.long
  )
  input_kwargs["attention_mask"] = ctx.ones(
      (ctx.batch_size, 8), dtype=torch.long
  )


def _build_s2t_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_mel = _safe_int(
      ctx.config.get("feature_projection_input_dim")
      or ctx.config.get("num_mel_bins"),
      default=80,
      min_val=1,
  )
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, 100, num_mel)
  input_kwargs["decoder_input_ids"] = ctx.full(
      (ctx.batch_size, 8), 1, dtype=torch.long
  )


def _build_univnet_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_mel = _cfg_int(ctx.config, "num_mel_bins", "input_dim", default=80)
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, 128, num_mel)


def _build_xcodec2_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  if ctx.shape and len(ctx.shape) > 1:
    seq_len = ctx.shape[1]
    input_kwargs["input_features"] = ctx.randn(ctx.batch_size, 1, seq_len)
    return

  channels = _cfg_int(ctx.config, "audio_channels", default=1)
  downsampling_ratios = ctx.config.get("downsampling_ratios") or [
      2,
      2,
      4,
      4,
      5,
  ]
  hop_length = math.prod(
      _safe_int(r, default=2, min_val=1) for r in downsampling_ratios
  )
  num_frames = 10
  num_samples = hop_length * num_frames
  sem_cfg = ctx.config.get("semantic_model_config")
  feat_dim = None
  if hasattr(sem_cfg, "feature_projection_input_dim"):
    feat_dim = getattr(sem_cfg, "feature_projection_input_dim", None)
  elif isinstance(sem_cfg, dict) and "feature_projection_input_dim" in sem_cfg:
    feat_dim = sem_cfg["feature_projection_input_dim"]
  if feat_dim is None:
    num_mel_bins = _cfg_int(ctx.config, "num_mel_bins", default=80)
    stride = _cfg_int(ctx.config, "stride", default=2)
    feat_dim = num_mel_bins * stride
  input_kwargs["input_values"] = ctx.randn(
      ctx.batch_size, channels, num_samples
  )
  input_kwargs["input_features"] = ctx.randn(
      ctx.batch_size, num_frames, feat_dim
  )


def _build_codec_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  seq_len = ctx.shape[1] if ctx.shape and len(ctx.shape) > 1 else 16000
  channels = _cfg_int(ctx.config, "audio_channels", default=1)
  input_kwargs["input_values"] = ctx.randn(ctx.batch_size, channels, seq_len)


def _build_musicgen_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 16)
  dec_cfg = ctx.config.get("decoder") or ctx.config
  num_codebooks = _cfg_int(dec_cfg, "num_codebooks", default=4)
  audio_vocab_size = _cfg_int(dec_cfg, "vocab_size", default=2048)
  if "decoder" in ctx.model_type or "causallm" in ctx.arch_name:
    input_kwargs["input_ids"] = ctx.randint(
        0, audio_vocab_size, (ctx.batch_size * num_codebooks, safe_seq_len)
    )
    return
  text_cfg = ctx.config.get("text_encoder") or ctx.config
  vocab_size = _cfg_int(text_cfg, "vocab_size", default=32000)
  input_kwargs.update(
      _build_text_tokens(ctx, seq_len=safe_seq_len, vocab_size=vocab_size)
  )
  input_kwargs["decoder_input_ids"] = ctx.randint(
      0, audio_vocab_size, (ctx.batch_size * num_codebooks, 1)
  )


def _build_bark_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 64)
  sem_cfg = (
      ctx.config.get("semantic_config")
      or ctx.config.get("text_config")
      or ctx.config
  )
  vocab_size = _cfg_int(
      sem_cfg, "vocab_size", "input_vocab_size", default=10048
  )
  input_kwargs.update(
      _build_text_tokens(ctx, seq_len=safe_seq_len, vocab_size=vocab_size)
  )


def _build_voxtral_realtime_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(
      ctx.shape[1]
      if ctx.shape and len(ctx.shape) > 1
      else _get_max_seq_len(ctx.config),
      32,
  )
  text_cfg = ctx.config.get("text_config") or ctx.config
  audio_cfg = ctx.config.get("audio_config") or ctx.config
  vocab_size = _safe_int(
      text_cfg.get("vocab_size") or ctx.config.get("vocab_size"),
      default=131072,
      min_val=1,
  )
  input_kwargs["input_ids"] = ctx.randint(
      0, vocab_size, (ctx.batch_size, safe_seq_len), dtype=torch.long
  )
  input_kwargs["attention_mask"] = ctx.ones(
      (ctx.batch_size, safe_seq_len), dtype=torch.long
  )
  num_mel = _safe_int(
      audio_cfg.get("num_mel_bins") or ctx.config.get("num_mel_bins"),
      default=128,
      min_val=1,
  )
  downsample_factor = _safe_int(
      ctx.config.get("downsample_factor"), default=4, min_val=1
  )
  audio_length_per_tok = _safe_int(
      ctx.config.get("audio_length_per_tok"),
      default=2 * downsample_factor,
      min_val=1,
  )
  feat_seq_len = safe_seq_len * audio_length_per_tok
  input_kwargs["input_features"] = ctx.randn(
      ctx.batch_size, num_mel, feat_seq_len
  )


def _build_csm_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 32)
  vocab_size = _cfg_int(ctx.config, "vocab_size", default=32000)
  input_kwargs.update(
      _build_text_tokens(ctx, seq_len=safe_seq_len, vocab_size=vocab_size)
  )


def _build_cohere_asr_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  enc_cfg = (
      ctx.config.get("encoder_config")
      or ctx.config.get("encoder")
      or ctx.config
  )
  num_mel = _safe_int(
      (enc_cfg.get("num_mel_bins") if enc_cfg else None)
      or ctx.config.get("num_mel_bins"),
      default=128,
      min_val=1,
  )
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, 128, num_mel)
  input_kwargs["decoder_input_ids"] = ctx.full(
      (ctx.batch_size, 8), 1, dtype=torch.long
  )


def _build_dia_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  enc_cfg = ctx.config.get("encoder_config") or ctx.config
  dec_cfg = ctx.config.get("decoder_config") or ctx.config
  enc_vocab = _safe_int(
      enc_cfg.get("vocab_size")
      if hasattr(enc_cfg, "get")
      else getattr(enc_cfg, "vocab_size", None),
      default=256,
      min_val=1,
  )
  dec_vocab = _safe_int(
      dec_cfg.get("vocab_size")
      if hasattr(dec_cfg, "get")
      else getattr(dec_cfg, "vocab_size", None),
      default=1028,
      min_val=1,
  )
  num_channels = _safe_int(
      dec_cfg.get("num_channels")
      if hasattr(dec_cfg, "get")
      else getattr(dec_cfg, "num_channels", None),
      default=9,
      min_val=1,
  )
  seq_len = 8
  input_kwargs["input_ids"] = ctx.randint(
      0, min(enc_vocab, 256), (ctx.batch_size, seq_len)
  )
  input_kwargs["attention_mask"] = ctx.ones((ctx.batch_size, seq_len))
  input_kwargs["decoder_input_ids"] = ctx.randint(
      1, dec_vocab, (ctx.batch_size, seq_len, num_channels)
  )


def _build_speecht5_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 8)
  vocab_size = _cfg_int(ctx.config, "vocab_size", default=81, min_val=4)
  max_vocab = vocab_size - 1
  num_mel = _cfg_int(ctx.config, "num_mel_bins", default=80)
  dec_seq_len = 8
  input_kwargs["decoder_attention_mask"] = ctx.ones(
      (ctx.batch_size, dec_seq_len)
  )
  if (
      "speechtotext" in ctx.arch_name
      or "speechtospeech" in ctx.arch_name
      or "asr" in ctx.type_str
  ):
    audio_seq_len = 16000
    input_kwargs["input_values"] = ctx.randn(ctx.batch_size, audio_seq_len)
    input_kwargs["attention_mask"] = ctx.ones((ctx.batch_size, audio_seq_len))
  else:
    input_kwargs.update(
        _build_text_tokens(
            ctx, seq_len=safe_seq_len, vocab_size=max_vocab, min_token_id=2
        )
    )

  if "speechtotext" in ctx.arch_name:
    input_kwargs["decoder_input_ids"] = ctx.randint(
        2, max_vocab, (ctx.batch_size, dec_seq_len)
    )
  else:
    input_kwargs["decoder_input_values"] = ctx.randn(
        ctx.batch_size, dec_seq_len, num_mel
    )
    speaker_dim = _cfg_int(ctx.config, "speaker_embedding_dim", default=512)
    input_kwargs["speaker_embeddings"] = ctx.randn(ctx.batch_size, speaker_dim)


def _build_vits_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 8)
  vocab_size = _cfg_int(ctx.config, "vocab_size", default=38, min_val=4)
  max_vocab = min(vocab_size, 38)
  input_kwargs.update(
      _build_text_tokens(
          ctx, seq_len=safe_seq_len, vocab_size=max_vocab, min_token_id=2
      )
  )
  # High speaking_rate prevents uninitialized to_empty log_duration overflow.
  input_kwargs["speaking_rate"] = 1e15


def _build_granite_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  encoder_cfg = (
      ctx.config.get("encoder_config")
      or ctx.config.get("encoder")
      or ctx.config.get("speech_config")
  )
  num_mel = _safe_int(
      (encoder_cfg.get("input_dim") or encoder_cfg.get("num_mel_bins"))
      if encoder_cfg
      else None,
      default=160,
      min_val=1,
  )
  window_size = _cfg_int(ctx.config, "window_size", default=15)
  downsample_rate = _cfg_int(ctx.config, "downsample_rate", default=5)
  audio_len = window_size  # nblocks = 1
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, audio_len, num_mel)
  # Calculate exact number of audio tokens expected by GraniteSpeech
  num_audio_tokens = window_size // downsample_rate
  audio_token_id = _safe_int(
      ctx.config.get("audio_token_id") or ctx.config.get("audio_token_index"),
      default=49152,
  )
  input_kwargs["input_ids"] = ctx.full(
      (ctx.batch_size, num_audio_tokens), audio_token_id
  )
  input_kwargs["attention_mask"] = ctx.ones((ctx.batch_size, num_audio_tokens))


def _build_fastspeech_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 16)
  model_cfg = ctx.config.get("model_config")
  vocab_size = _safe_int(
      ctx.config.get("vocab_size")
      or (model_cfg.get("vocab_size") if model_cfg else None),
      default=38,
      min_val=1,
  )
  input_kwargs.update(
      _build_text_tokens(ctx, seq_len=safe_seq_len, vocab_size=vocab_size)
  )


def _build_pe_audio_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  safe_seq_len = min(_get_max_seq_len(ctx.config), 16)
  text_cfg = ctx.config.get("text_config")
  vocab_size = _safe_int(
      ctx.config.get("vocab_size")
      or (text_cfg.get("vocab_size") if text_cfg else None),
      default=32000,
      min_val=1,
  )
  input_kwargs.update(
      _build_text_tokens(ctx, seq_len=safe_seq_len, vocab_size=vocab_size)
  )
  audio_cfg = ctx.config.get("audio_config")
  dac_cfg = audio_cfg.get("dac_config") if audio_cfg else None
  hop_length = _cfg_int(dac_cfg, "hop_length", default=1920)
  audio_len = safe_seq_len * hop_length
  input_kwargs["input_values"] = ctx.randn(ctx.batch_size, 1, audio_len)


def _build_qwen_omni_audio_encoder_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_mel = _cfg_int(ctx.config, "num_mel_bins", "feature_size", default=128)
  if ctx.shape and len(ctx.shape) > 1:
    frames_per_sample = ctx.shape[1]
  else:
    n_window = _cfg_int(ctx.config, "n_window", default=50)
    frames_per_sample = n_window * 2
  input_kwargs["input_features"] = ctx.randn(
      num_mel, ctx.batch_size * frames_per_sample
  )
  input_kwargs["feature_lens"] = ctx.full(
      (ctx.batch_size,), frames_per_sample, dtype=torch.long
  )


_AUDIO_EXTRA_BUILDERS: tuple[tuple[_Matcher, _Builder], ...] = (
    (_model_type_contains("whisper"), _build_whisper_audio_inputs),
    (_model_type_contains("clap"), _build_clap_audio_inputs),
    (
        _type_contains("audio-spectrogram-transformer", "astmodel", "astfor"),
        _build_ast_audio_inputs,
    ),
    (
        _type_contains("fastspeech", "fastspeech2"),
        _build_fastspeech_audio_inputs,
    ),
    (
        _type_contains(
            "qwen2_5_omni_audio_encoder",
            "qwen3_omni_moe_audio_encoder",
            "qwen3omnimoeaudioencoder",
            "qwen2_5omniaudioencoder",
        ),
        _build_qwen_omni_audio_encoder_inputs,
    ),
    (
        _type_contains(
            "w2v-bert",
            "wav2vec2-bert",
            "lasr_ctc",
            "parakeet",
            "nemotron",
            "gemma4_audio",
            "gemma4audio",
            "qwen2_audio",
            "qwen2-audio",
            "qwen2audio",
            "qwen2a",
        ),
        _build_w2v_bert_audio_inputs,
    ),
    (_type_contains("gemma3n", "usm"), _build_gemma3n_audio_inputs),
    (
        _type_contains("omnitoken2wav", "scail"),
        _build_omnitoken2wav_audio_inputs,
    ),
    (_model_type_contains("speecht5"), _build_speecht5_audio_inputs),
    (
        _type_contains("kyutai_speech_to_text", "kyutai_stt"),
        _build_kyutai_stt_inputs,
    ),
    (_type_contains("moshi"), _build_moshi_audio_inputs),
    (
        _type_contains(
            "glmasr",
            "glm_asr",
            "glm-asr",
            "qwen3_asr",
            "qwen3-asr",
            "qwen3asr",
            "vibevoice_asr",
            "vibevoice-asr",
            "vibevoiceasr",
        ),
        _build_glmasr_audio_inputs,
    ),
    (
        _type_contains("cohere_asr", "cohere-asr"),
        _build_cohere_asr_audio_inputs,
    ),
    (
        _type_contains("dia"),
        _build_dia_audio_inputs,
    ),
    (
        _type_contains(
            "s2t",
            "speech2text",
            "speech_to_text",
            "seamless",
            "speechtotext",
        ),
        _build_s2t_audio_inputs,
    ),
    (_type_contains("univnet"), _build_univnet_audio_inputs),
    (_type_contains("xcodec2"), _build_xcodec2_audio_inputs),
    (
        _type_contains(
            "dac",
            "encodec",
            "higgs_audio_v2_tokenizer",
            "mimi",
            "vibevoice_acoustic_tokenizer",
            "xcodec",
        ),
        _build_codec_audio_inputs,
    ),
    (_type_contains("musicgen"), _build_musicgen_audio_inputs),
    (_type_contains("bark"), _build_bark_audio_inputs),
    (
        _type_contains("voxtral_realtime", "voxtralrealtime"),
        _build_voxtral_realtime_audio_inputs,
    ),
    (
        _type_contains(
            "csm", "voxtral", "dia", "audio2hero", "higgs", "pop2piano"
        ),
        _build_csm_audio_inputs,
    ),
    (_model_type_contains("vits"), _build_vits_audio_inputs),
    (_model_type_contains("granite"), _build_granite_audio_inputs),
    (
        _type_contains("pe_audio", "pe-audio", "pe-a-frame"),
        _build_pe_audio_inputs,
    ),
)


def _min_feature_encoder_samples(config: Any) -> int:
  """Returns the fewest waveform samples that the feature encoder accepts.

  Speech encoders such as wav2vec2, HuBERT and WavLM turn the raw waveform into
  frames with a stack of strided 1D convolutions (`conv_kernel` and
  `conv_stride`), so a waveform shorter than the receptive field of one frame
  leaves a convolution with fewer inputs than its kernel.

  Args:
    config: The model config.

  Returns:
    The receptive field of one frame, or 0 if the model has no convolutional
    feature encoder.
  """
  kernels = _cfg_lookup(config, "conv_kernel")
  strides = _cfg_lookup(config, "conv_stride")
  if not isinstance(kernels, Sequence) or not isinstance(strides, Sequence):
    return 0
  samples = 1
  for kernel, stride in zip(reversed(kernels), reversed(strides)):
    samples = (samples - 1) * stride + kernel
  return samples


def _build_audio_inputs(ctx: InputContext) -> dict[str, Any]:
  """Builds inputs for audio models."""
  input_kwargs: dict[str, Any] = {}
  _apply_builders(
      ctx, input_kwargs, _AUDIO_EXTRA_BUILDERS, first_match_only=True
  )
  if not input_kwargs:
    seq_len = ctx.shape[1] if ctx.shape and len(ctx.shape) > 1 else 16000
    # Short shapes, e.g. the 128 steps of the perflab presubmit run scope, are
    # too short a waveform for a convolutional feature encoder.
    seq_len = max(seq_len, _min_feature_encoder_samples(ctx.config))
    input_kwargs["input_values"] = ctx.randn(ctx.batch_size, seq_len)
  return input_kwargs


# ---------------------------------------------------------------------------
# Multimodal Modality Builders
# ---------------------------------------------------------------------------


def _build_idefics_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  is_video = (
      "video" in str(getattr(ctx.config, "name_or_path", "")).lower()
      or "video" in ctx.type_str
  )
  num_images = 16 if is_video else 1
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size, num_images, num_channels, image_size, image_size
  )
  if "idefics2" in ctx.model_type or "smolvlm" in ctx.model_type:
    input_kwargs["pixel_attention_mask"] = torch.ones(
        (ctx.batch_size, num_images, image_size, image_size),
        device=ctx.device,
        dtype=torch.bool,
    )
  elif "idefics" in ctx.model_type:
    input_kwargs["image_attention_mask"] = torch.ones(
        (ctx.batch_size, input_kwargs["input_ids"].shape[1], 1),
        device=ctx.device,
        dtype=torch.bool,
    )


def _build_video_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  num_frames = _cfg_int(
      ctx.config, "vision_config.num_frames", "num_frames", default=8
  )
  input_kwargs["pixel_values_videos"] = ctx.randn(
      ctx.batch_size, num_frames, num_channels, image_size, image_size
  )
  input_kwargs.pop("pixel_values", None)


def _build_llava_next_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  size_device = "cpu" if str(ctx.device) == "meta" else ctx.device
  input_kwargs["image_sizes"] = torch.tensor(
      [[image_size, image_size]] * ctx.batch_size,
      device=size_device,
      dtype=torch.long,
  )


def _build_minicpmv_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_channels = _get_num_channels(ctx.config)
  vision_cfg = ctx.config.get("vision_config")
  patch_size = _cfg_int(vision_cfg, "patch_size", default=14)
  grid_side = 4
  img_size = grid_side * patch_size
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size, num_channels, img_size, img_size
  )
  input_kwargs["target_sizes"] = torch.tensor(
      [[grid_side, grid_side]] * ctx.batch_size,
      device=ctx.device,
      dtype=torch.int32,
  )


def _build_qformer_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  num_query_tokens = _cfg_int(ctx.config, "num_query_tokens", default=32)
  hidden_size = _cfg_int(ctx.config, "hidden_size", default=768)
  encoder_hidden_size = _cfg_int(
      ctx.config, "encoder_hidden_size", default=hidden_size
  )
  input_kwargs["query_embeds"] = ctx.randn(
      ctx.batch_size, num_query_tokens, hidden_size
  )
  input_kwargs["encoder_hidden_states"] = ctx.randn(
      ctx.batch_size, 32, encoder_hidden_size
  )


def _build_instructblip_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["qformer_input_ids"] = input_kwargs["input_ids"]
  input_kwargs["qformer_attention_mask"] = input_kwargs["attention_mask"]


def _build_blip_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["decoder_input_ids"] = torch.tensor(
      [[1]], device=ctx.device, dtype=torch.long
  )


def _build_lxmert_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["visual_feats"] = ctx.randn(ctx.batch_size, 10, 2048)
  input_kwargs["visual_pos"] = ctx.zeros(
      (ctx.batch_size, 10, 4), dtype=torch.float32
  )


def _build_udop_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs["bbox"] = ctx.zeros(
      (*input_kwargs["input_ids"].shape, 4), dtype=torch.long
  )
  input_kwargs["decoder_input_ids"] = torch.tensor(
      [[1]], device=ctx.device, dtype=torch.long
  )


def _build_pix2struct_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  max_patches = min(
      _cfg_int(ctx.config, "max_patches", default=2048),
      32,
  )
  vision_cfg = ctx.config.get("vision_config") or ctx.config
  text_cfg = ctx.config.get("text_config") or ctx.config
  patch_hidden_size = _safe_int(
      vision_cfg.get("patch_embed_hidden_size")
      or ctx.config.get("patch_embed_hidden_size")
      or text_cfg.get("hidden_size"),
      default=768,
      min_val=1,
  )
  input_kwargs.pop("pixel_values", None)
  input_kwargs["flattened_patches"] = ctx.randn(
      ctx.batch_size, max_patches, patch_hidden_size + 2
  )
  input_kwargs["attention_mask"] = ctx.ones(
      (ctx.batch_size, max_patches), dtype=torch.long
  )
  input_kwargs["decoder_input_ids"] = torch.tensor(
      [[1]], device=ctx.device, dtype=torch.long
  )


def _build_qwen_vl_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  num_img_tokens = _build_spatial_merged_patches(
      ctx, input_kwargs, image_size=image_size, num_channels=num_channels
  )
  img_token_id = _cfg_int(ctx.config, "image_token_id", default=151652)
  text_cfg = ctx.config.get("text_config")
  vocab_size = _safe_int(
      ctx.config.get("src_vocab_size")
      or ctx.config.get("vocab_size")
      or (text_cfg.get("vocab_size") if text_cfg else None),
      default=32000,
      min_val=1,
  )
  cur_seq_len = _ensure_min_seq_len(
      ctx,
      input_kwargs,
      needed_len=num_img_tokens + 4,
      vocab_size=min(vocab_size, 50000),
  )
  valid_tokens = min(num_img_tokens, max(0, cur_seq_len - 1))
  if valid_tokens > 0:
    _inject_token_span(
        input_kwargs["input_ids"], img_token_id, start=1, count=valid_tokens
    )
    input_kwargs["mm_token_type_ids"] = _mm_token_type_ids(
        input_kwargs["input_ids"], start=1, count=valid_tokens
    )


def _build_glm_vl_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  num_img_tokens = _build_spatial_merged_patches(
      ctx, input_kwargs, image_size=image_size, num_channels=num_channels
  )
  img_start_id = _cfg_int(ctx.config, "image_start_token_id", default=59256)
  img_token_id = _cfg_int(ctx.config, "image_token_id", default=59280)
  img_end_id = _cfg_int(ctx.config, "image_end_token_id", default=59257)

  text_cfg = ctx.config.get("text_config")
  vocab_size = _safe_int(
      ctx.config.get("src_vocab_size")
      or ctx.config.get("vocab_size")
      or (text_cfg.get("vocab_size") if text_cfg else None),
      default=32000,
      min_val=1,
  )
  cur_seq_len = _ensure_min_seq_len(
      ctx,
      input_kwargs,
      needed_len=num_img_tokens + 4,
      vocab_size=min(vocab_size, 50000),
  )
  valid_tokens = min(num_img_tokens, max(0, cur_seq_len - 2))
  if cur_seq_len > 0:
    input_kwargs["input_ids"][:, 0] = img_start_id
  if valid_tokens > 0:
    _inject_token_span(
        input_kwargs["input_ids"], img_token_id, start=1, count=valid_tokens
    )
  if 1 + valid_tokens < cur_seq_len:
    input_kwargs["input_ids"][:, 1 + valid_tokens] = img_end_id

  if valid_tokens > 0:
    input_kwargs["mm_token_type_ids"] = _mm_token_type_ids(
        input_kwargs["input_ids"], start=1, count=valid_tokens
    )
  else:
    input_kwargs["mm_token_type_ids"] = torch.zeros_like(
        input_kwargs["input_ids"], dtype=torch.int
    )


def _build_kosmos_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  latent_query_num = _cfg_int(ctx.config, "latent_query_num", default=64)
  text_cfg = ctx.config.get("text_config")
  vocab_size = _safe_int(
      ctx.config.get("src_vocab_size")
      or ctx.config.get("vocab_size")
      or (text_cfg.get("vocab_size") if text_cfg else None),
      default=32000,
      min_val=1,
  )
  cur_seq_len = _ensure_min_seq_len(
      ctx,
      input_kwargs,
      needed_len=latent_query_num + 2,
      vocab_size=vocab_size,
      target_len=latent_query_num + 16,
  )
  pos_mask = ctx.zeros((ctx.batch_size, cur_seq_len))
  pos_mask[:, 1 : 1 + latent_query_num] = 1
  input_kwargs["image_embeds_position_mask"] = pos_mask

  if "kosmos-2.5" in ctx.model_type or "kosmos_2_5" in ctx.model_type:
    vision_config = ctx.config.get("vision_config")
    patch_size = _cfg_int(vision_config, "patch_size", default=16)
    image_size = _parse_image_size(ctx.config)
    num_channels = _get_num_channels(ctx.config)
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size * num_channels + 2
    input_kwargs["flattened_patches"] = ctx.randn(
        ctx.batch_size, num_patches + 1, patch_dim
    )
    input_kwargs.pop("pixel_values", None)


def _build_xclip_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  image_size = _parse_image_size(ctx.config)
  num_channels = _get_num_channels(ctx.config)
  num_frames = _cfg_int(
      ctx.config, "vision_config.num_frames", "num_frames", default=8
  )
  input_kwargs["pixel_values"] = ctx.randn(
      ctx.batch_size, num_frames, num_channels, image_size, image_size
  )


def _build_vilt_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  # ViLT selects its image patches with data-dependent ops on `pixel_mask`.
  # Without one, it makes a floating-point `torch.ones` mask, whose values the
  # FakeTensorMode CPU smoke test does not keep, so pass the integer mask that
  # `ViltImageProcessor` returns.
  _, _, height, width = input_kwargs["pixel_values"].shape
  input_kwargs["pixel_mask"] = ctx.ones((ctx.batch_size, height, width))


def _build_beingvl_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs.pop("input_ids", None)
  input_kwargs.pop("attention_mask", None)


def _build_clvp_multimodal_inputs(
    ctx: InputContext, input_kwargs: dict[str, Any]
) -> None:
  input_kwargs.pop("pixel_values", None)
  dec_cfg = ctx.config.get("decoder_config") or ctx.config
  feature_size = _safe_int(
      dec_cfg.get("feature_size") if dec_cfg else None,
      default=80,
      min_val=1,
  )
  input_kwargs["input_features"] = ctx.randn(ctx.batch_size, feature_size, 100)


_MULTIMODAL_EXTRA_BUILDERS: tuple[tuple[_Matcher, _Builder], ...] = (
    (
        _model_type_contains("idefics", "smolvlm"),
        _build_idefics_multimodal_inputs,
    ),
    (
        _model_type_contains("videoprism", "videomt"),
        _build_video_multimodal_inputs,
    ),
    (
        _model_type_contains(
            "llava_next", "llava_onevision", "llava-v1.6", "llava-next"
        ),
        _build_llava_next_multimodal_inputs,
    ),
    (_model_type_contains("qformer"), _build_qformer_multimodal_inputs),
    (_model_type_contains("oneformer"), _build_oneformer_inputs),
    (
        _model_type_contains("instructblip"),
        _build_instructblip_multimodal_inputs,
    ),
    (_model_type_contains("blip"), _build_blip_multimodal_inputs),
    (_model_type_contains("lxmert"), _build_lxmert_multimodal_inputs),
    (_model_type_contains("udop"), _build_udop_multimodal_inputs),
    (_model_type_contains("pix2struct"), _build_pix2struct_multimodal_inputs),
    (
        lambda ctx: any(
            k in ctx.model_type for k in _PATCHIFIED_VLM_MODEL_TYPES
        )
        and not any(k in ctx.model_type for k in ("glm_ocr", "glm4v")),
        _build_qwen_vl_multimodal_inputs,
    ),
    (
        _model_type_contains("glm_ocr", "glm4v"),
        _build_glm_vl_multimodal_inputs,
    ),
    (_model_type_contains("kosmos"), _build_kosmos_multimodal_inputs),
    (_model_type_contains("xclip"), _build_xclip_multimodal_inputs),
    (_model_type_contains("vilt"), _build_vilt_multimodal_inputs),
    (_model_type_contains("minicpmv"), _build_minicpmv_multimodal_inputs),
    (
        _model_type_contains("flava"),
        lambda ctx, kw: kw.__setitem__("return_loss", False),
    ),
    (
        _model_type_contains("emu3", "chameleon"),
        lambda ctx, kw: kw.pop("pixel_values", None),
    ),
    (_model_type_contains("beingvl", "vq"), _build_beingvl_multimodal_inputs),
    (_type_contains("clvp"), _build_clvp_multimodal_inputs),
    (
        _type_contains("siglip2"),
        lambda ctx, kw: _build_siglip2_inputs(ctx, kw),
    ),
)


def _compute_num_image_tokens(
    ctx: InputContext,
    image_size: int,
    vision_config: Any,
) -> int | None:
  """Computes the expected visual token count injected into input_ids."""
  if "ovis" in ctx.model_type:
    return 256
  if "minicpmv" in ctx.model_type:
    return 1
  if "blip-2" in ctx.model_type or "blip_2" in ctx.model_type:
    return _cfg_int(ctx.config, "num_query_tokens", default=32)
  if "idefics2" in ctx.model_type:
    return _cfg_int(
        ctx.config, "perceiver_config.resampler_n_latents", default=64
    )
  if "fast_vlm" in ctx.model_type:
    return (image_size // 64) ** 2
  if any(k in ctx.model_type for k in ("llava_onevision", "llava_next")):
    patch_size = _cfg_int(vision_config, "patch_size", default=14)
    h = image_size // patch_size
    return 2 * (h * h) + h
  if any(k in ctx.type_str for k in ("smolvlm", "idefics3")):
    patch_size = _cfg_int(vision_config, "patch_size", default=16)
    scale = _cfg_int(ctx.config, "scale_factor", default=4)
    is_video = (
        "video" in str(getattr(ctx.config, "name_or_path", "")).lower()
        or "video" in ctx.type_str
    )
    num_frames = 16 if is_video else 1
    return num_frames * (((image_size // patch_size) // scale) ** 2)
  if "got_ocr2" in ctx.model_type:
    patch_size = _safe_int(
        vision_config.get("patch_size") if vision_config else None,
        default=16,
        min_val=1,
    )
    return ((image_size // patch_size) // 4) ** 2
  vlm_cfg = ctx.config.get("vlm_config")
  raw_tokens = _cfg_int(
      ctx.config,
      "image_seq_length",
      "vlm_config.image_seq_length",
      "vlm_config.num_image_tokens",
      default=None,
  ) or (
      _cfg_int(vlm_cfg, "num_image_tokens", default=None) if vlm_cfg else None
  )
  if raw_tokens is not None and not (
      "llava" in ctx.model_type and raw_tokens == 1
  ):
    return raw_tokens
  if "florence2" in ctx.model_type:
    return (image_size // 32) ** 2 + 1
  patch_size = _cfg_int(vision_config, "patch_size", default=14)
  return (image_size // patch_size) ** 2


def _build_multimodal_inputs(ctx: InputContext) -> dict[str, Any]:
  """Builds inputs for multimodal models."""
  max_pos = _get_max_seq_len(ctx.config)
  safe_seq_len = min(max_pos, 512)
  if ctx.shape is not None:
    actual_shape = (
        ctx.shape[0],
        min(ctx.shape[1], max_pos) if len(ctx.shape) > 1 else safe_seq_len,
    )
  else:
    actual_shape = (1, safe_seq_len)
  # Check `src_vocab_size` before `vocab_size` for encoder-decoder models
  # with asymmetric source and target vocabularies such as FSMT
  # (`facebook/wmt19-ru-en`), where `FSMTConfig.attribute_map` aliases
  # `config.vocab_size` to `config.tgt_vocab_size` (31640) while the encoder
  # embedding table (`FSMTEncoder.embed_tokens`) has size `src_vocab_size`
  # (31232). Sampling `input_ids` up to `vocab_size` would otherwise produce
  # out-of-bounds token IDs in `[31232, 31640)` for the encoder on CUDA.
  vlm_cfg = ctx.config.get("vlm_config")
  thinker_cfg = ctx.config.get("thinker_config")
  text_cfg = (
      ctx.config.get("text_config")
      or (vlm_cfg.get("text_config") if vlm_cfg else None)
      or (thinker_cfg.get("text_config") if thinker_cfg else None)
      or thinker_cfg
  )
  semantic_cfg = ctx.config.get("semantic_config")
  qformer_cfg = ctx.config.get("qformer_config")
  if ("retrieval" in ctx.arch_name or "itm" in ctx.type_str) and qformer_cfg:
    vocab_size = _safe_int(
        qformer_cfg.get("vocab_size"), default=30522, min_val=1
    )
  else:
    vocab_size = _safe_int(
        ctx.config.get("src_vocab_size")
        or (text_cfg.get("vocab_size") if text_cfg else None)
        or ctx.config.get("vocab_size")
        or (semantic_cfg.get("input_vocab_size") if semantic_cfg else None),
        default=32000,
        min_val=1,
    )

  input_kwargs = {
      "input_ids": ctx.randint(0, vocab_size, actual_shape),
      "attention_mask": ctx.ones(actual_shape),
  }

  if ctx.model_type.startswith("gemma4"):
    image_size = _parse_image_size(ctx.config, default_size=288)
    _generate_gemma4_inputs(
        ctx.config,
        actual_shape[0],
        image_size,
        ctx.device,
        input_kwargs,
        generator=ctx.generator,
    )
  elif any(k in ctx.model_type for k in _VISION_LANGUAGE_MODEL_TYPES):
    image_size = _parse_image_size(ctx.config)
    if any(k in ctx.type_str for k in ["smolvlm", "idefics3"]):
      image_size = 512
    num_channels = _get_num_channels(ctx.config)
    batch_size = ctx.batch_size
    vision_config = ctx.config.get("vision_config") or (
        vlm_cfg.get("vision_config") if vlm_cfg else None
    )

    if "mllama" in ctx.model_type:
      num_images = 1
      num_tiles = _cfg_int(vision_config, "max_num_tiles", default=4)
      dummy_img = ctx.randn(
          batch_size,
          num_images,
          num_tiles,
          num_channels,
          image_size,
          image_size,
      )
      input_kwargs["pixel_values"] = dummy_img
      input_kwargs["aspect_ratio_ids"] = ctx.ones((batch_size, num_images))
      input_kwargs["aspect_ratio_mask"] = ctx.ones(
          (batch_size, num_images, num_tiles)
      )
    else:
      if "llava_onevision" in ctx.model_type:
        input_kwargs["pixel_values"] = ctx.randn(
            batch_size, 2, num_channels, image_size, image_size
        )
      elif "llava_next" in ctx.model_type:
        input_kwargs["pixel_values"] = ctx.randn(
            batch_size, 3, num_channels, image_size, image_size
        )
      elif "tvp" in ctx.type_str:
        input_kwargs["pixel_values"] = ctx.randn(
            batch_size, 1, num_channels, 448, 448
        )
      else:
        dummy_img = ctx.randn(
            batch_size,
            num_channels,
            image_size,
            image_size,
        )
        input_kwargs["pixel_values"] = dummy_img

      if any(k in ctx.type_str for k in ["videoprism", "videomt"]):
        num_frames = 32
        img_s = 288
        input_kwargs["pixel_values_videos"] = ctx.randn(
            batch_size, num_frames, num_channels, img_s, img_s
        )
        input_kwargs.pop("pixel_values", None)

      is_dual_encoder = any(
          k in ctx.model_type for k in _DUAL_ENCODER_MODEL_TYPES
      )
      if (
          not is_dual_encoder
          and "pix2struct" not in ctx.model_type
          and "instructblip" not in ctx.model_type
          and "git" not in ctx.model_type
          and not any(
              k in ctx.model_type for k in _LEGACY_NO_INJECT_VLM_MODEL_TYPES
          )
      ):
        num_image_tokens = _compute_num_image_tokens(
            ctx, image_size, vision_config
        )

        image_token_id = (
            ctx.config.get("image_token_id")
            or ctx.config.get("image_token_index")
            or (
                vlm_cfg.get("image_token_id")
                or vlm_cfg.get("image_token_index")
                if vlm_cfg
                else None
            )
        )
        if image_token_id is not None:
          image_token_id = _safe_int(image_token_id, default=None)

        if image_token_id is not None and num_image_tokens is not None:
          _ensure_min_seq_len(
              ctx,
              input_kwargs,
              needed_len=num_image_tokens,
              vocab_size=vocab_size,
              target_len=num_image_tokens + 16,
          )
          _inject_token_span(
              input_kwargs["input_ids"],
              image_token_id,
              start=0,
              count=num_image_tokens,
          )

    _apply_builders(
        ctx, input_kwargs, _MULTIMODAL_EXTRA_BUILDERS, first_match_only=False
    )

  return input_kwargs


_INPUT_BUILDERS_BY_MODALITY = {
    "text": lambda ctx, md, pl: _build_text_inputs(ctx),
    "causal_lm": lambda ctx, md, pl: _build_text_inputs(ctx),
    "seq2seq": lambda ctx, md, pl: _build_text_inputs(ctx),
    "vision": lambda ctx, md, pl: _build_vision_inputs(ctx, md, pl),
    "audio": lambda ctx, md, pl: _build_audio_inputs(ctx),
    "multimodal": lambda ctx, md, pl: _build_multimodal_inputs(ctx),
}


def generate_transformers_inputs(
    config: Any,
    modality: Any,
    shape: Sequence[int] | None = None,
    device: str = "cpu",
    model_dir: str | None = None,
    preprocessor_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
  """Generates dummy inputs for a Transformers model based on its modality.

  Args:
    config: The Transformers configuration object.
    modality: The model modality (enum or string).
    shape: Optional input shape override.
    device: The target device for the inputs.
    model_dir: Optional local path to the model directory (for loading
      processor).
    preprocessor_loader: Optional callable taking `model_dir` and returning a
      loaded processor.

  Returns:
    A dictionary of input tensors.
  """
  ctx = InputContext.create(config, shape, device)
  mod_key = (
      modality.value if hasattr(modality, "value") else str(modality).lower()
  )
  if "." in mod_key:
    mod_key = mod_key.split(".")[-1]
  builder = _INPUT_BUILDERS_BY_MODALITY.get(
      mod_key, lambda c, md, pl: _build_text_inputs(c)
  )
  input_kwargs = builder(ctx, model_dir, preprocessor_loader)
  _add_encoder_decoder_inputs(ctx, input_kwargs, mod_key)
  return _cast_floating_point_inputs(ctx.config, input_kwargs)
