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

"""Qwen3-VL quality benchmark model."""

from typing import Any
import torch
from examples.benchmarks.e2e import device_utils
from examples.benchmarks.quality_utils import quality_benchmark_model
from tests import module_registry


class Qwen3VLQualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """Qwen3-VL quality benchmark model.

  Attributes:
    _model: The Qwen3-VL model.
  """

  def __init__(self, model_name: str, device: str):
    super().__init__()
    self._model_name = model_name
    self._device_str = device
    self._device = None
    self._model = None
    self._registry = module_registry.ModuleRegistry()
    self._module_spec = None

  @property
  def max_seq_len(self) -> int:
    """The maximum sequence length supported by the model."""
    return 512

  def encode(self, text: str) -> list[int]:
    """Encodes the text into tokens without padding or truncation."""
    assert self._module_spec is not None
    config = self._module_spec.config

    vocab_size = (
        config.text_config.vocab_size
        if hasattr(config, "text_config")
        else config.vocab_size
    )

    return [ord(c) % vocab_size for c in text]

  def initialize(self) -> None:
    """Initializes the model."""
    if self._device_str == "tpu":
      try:
        from torch_tpu import _loader

        _loader.load()
        self._device = torch.device("tpu")
      except (RuntimeError, ImportError):
        import logging as std_logging

        std_logging.warning(
            "TPU device not registered in model. Falling back to CPU."
        )
        self._device = torch.device("cpu")
    else:
      self._device = torch.device(self._device_str)

    self._module_spec = self._registry.get_module_spec(
        "transformers", self._model_name, load_weights=False
    )
    config = self._module_spec.config
    dtype = getattr(config, "torch_dtype", None) or getattr(
        config, "dtype", torch.float32
    )
    if isinstance(dtype, str):
      if dtype == "bfloat16":
        dtype = torch.bfloat16
      elif dtype == "float16":
        dtype = torch.float16
      elif dtype == "float32":
        dtype = torch.float32
      else:
        dtype = torch.float32
    elif not isinstance(dtype, torch.dtype):
      dtype = torch.float32

    self._model = self._module_spec.module_factory().to(self._device).to(dtype)
    self._model.eval()

  def get_model(self) -> torch.nn.Module:
    """Gets the model."""
    return self._model

  def _compile_model_once(self) -> None:
    """Compiles the model for the target device."""
    assert self._model is not None
    self._model = device_utils.torch_compile(self._model, self._device.type)

  def format(self, raw_input: Any) -> quality_benchmark_model.FormattedInput:
    """Encodes raw input for Qwen3-VL model.

    Uses a mock tokenization mapping characters to token IDs to enable offline
    pipeline verification without downloading the real tokenizer.

    Args:
      raw_input: The text (str) or list of tokens (list[int]) to encode.

    Returns:
      The encoded text as a tokens tensor and the length of the tokens.
    """
    assert self._module_spec is not None
    config = self._module_spec.config

    vocab_size = (
        config.text_config.vocab_size
        if hasattr(config, "text_config")
        else config.vocab_size
    )

    if isinstance(raw_input, str):
      tokens = [ord(c) % vocab_size for c in raw_input]
    elif isinstance(raw_input, (list, tuple)):
      tokens = list(raw_input)
    else:
      raise TypeError(f"Unsupported input type: {type(raw_input)}")
    max_len = 512
    tokens = tokens[:max_len]

    # We must make sure the sequence length is at least 2 for next-token prediction
    if len(tokens) < 2:
      tokens = [0, 0]

    tokens_tensor = torch.tensor(
        [tokens], dtype=torch.long, device=self._device
    )
    return quality_benchmark_model.FormattedInput(tokens_tensor, len(tokens))

  def get_logits_and_targets(
      self, formatted_input: quality_benchmark_model.FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns logits and targets aligned for loss calculation."""
    assert self._model is not None

    # Forward through the model (next-token prediction)
    input_ids = formatted_input.input[:, :-1]
    outputs = self._model(input_ids=input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

    # Remove batch dimension (batch_size=1)
    pred_logits = logits[0]

    # Target is shifted by 1
    target = formatted_input.input[0, 1:]

    return pred_logits, target


import dataclasses


@dataclasses.dataclass
class VLFormattedInput(quality_benchmark_model.FormattedInput):
  pixel_values: torch.Tensor | None = None
  image_grid_thw: torch.Tensor | None = None
  mm_token_type_ids: torch.Tensor | None = None


class DistributedQwen3VLQualityBenchmarkModel(
    quality_benchmark_model.QualityBenchmarkModel
):
  """Distributed Qwen3-VL quality benchmark model using FSDP and real weights/tokenizer.

  Attributes:
    _model: The sharded Qwen3-VL model.
  """

  def __init__(
      self,
      model_name: str,
      device: str,
      max_seq_len: int = 512,
  ):
    super().__init__()
    self._model_name = model_name
    self._device_str = device
    self._device = None
    self._model = None
    self._tokenizer = None
    self._max_seq_len = max_seq_len

  def initialize(self) -> None:
    """Initializes the model and tokenizer/processor."""
    from examples.benchmarks.e2e import model_utils
    from examples.benchmarks.e2e import performance_utils

    if self._device_str == "tpu":
      from torch_tpu import _loader

      _loader.load()
    self._device = torch.device(self._device_str)

    model_args = performance_utils.ModelAndInputArgs(
        model_name=self._model_name,
        sequence_length=self._max_seq_len,
        batch_size=1,
        custom_kwargs={
            "dist_strat": "fsdp",
            "load_weights": True,
        },
    )
    model_and_input = model_utils.huggingface_llm_model_builder(
        model_and_input_args=model_args,
        device=self._device,
        weights_dtype=torch.bfloat16,
        is_training=False,
    )
    self._model = model_and_input.model

    # Load preprocessor/tokenizer
    registry = model_utils.get_module_registry()
    module_spec = registry.get_module_spec(
        "transformers", self._model_name, load_weights=True
    )
    try:
      import transformers

      self._tokenizer = transformers.AutoProcessor.from_pretrained(
          self._model_name
      )
    except Exception:
      self._tokenizer = module_spec.preprocessor_factory()

  @property
  def max_seq_len(self) -> int:
    return self._max_seq_len

  def get_model(self) -> torch.nn.Module:
    return self._model

  def encode(self, text: str) -> list[int]:
    assert self._tokenizer is not None, "Tokenizer not initialized"
    if hasattr(self._tokenizer, "encode"):
      return self._tokenizer.encode(text, add_special_tokens=True)
    elif hasattr(self._tokenizer, "tokenizer"):
      return self._tokenizer.tokenizer.encode(text, add_special_tokens=True)
    else:
      raise AttributeError("No encode method found in tokenizer/processor")

  def format(self, raw_input: Any) -> VLFormattedInput:
    """Encodes raw input with vision features if present."""
    import json
    import os

    assert self._tokenizer is not None

    pixel_values = None
    image_grid_thw = None
    mm_token_type_ids = None

    if isinstance(raw_input, str):
      try:
        data = json.loads(raw_input)
        if isinstance(data, dict) and "image_path" in data:
          image_path = data.get("image_path")
          question = data.get("question", "")

          from PIL import Image

          try:
            if os.path.exists(image_path):
              image = Image.open(image_path)
            else:
              image = Image.new("RGB", (224, 224), color="red")
          except Exception:
            image = Image.new("RGB", (224, 224), color="red")

          inputs = self._tokenizer(
              text=[question],
              images=[image],
              padding=True,
              return_tensors="pt",
          )

          input_ids = inputs["input_ids"].to(self._device)
          pixel_values = inputs["pixel_values"].to(self._device)
          image_grid_thw = inputs["image_grid_thw"].to(self._device)
          if "mm_token_type_ids" in inputs:
            mm_token_type_ids = inputs["mm_token_type_ids"].to(self._device)
          else:
            image_token_id = getattr(self._tokenizer, "image_token_id", 151851)
            mm_token_type_ids = (
                (input_ids == image_token_id).long().to(self._device)
            )

          total_len = self._max_seq_len + 1
          tokens_len = input_ids.size(1)
          if tokens_len < total_len:
            pad_id = (
                self._tokenizer.pad_token_id
                if getattr(self._tokenizer, "pad_token_id", None) is not None
                else getattr(self._tokenizer, "eos_token_id", 0)
            )
            padded_ids = torch.full(
                (1, total_len), pad_id, dtype=torch.long, device=self._device
            )
            padded_ids[0, :tokens_len] = input_ids[0]
            input_ids = padded_ids
          else:
            input_ids = input_ids[:, :total_len]
            tokens_len = total_len

          return VLFormattedInput(
              input=input_ids,
              unpadded_length=tokens_len,
              pixel_values=pixel_values,
              image_grid_thw=image_grid_thw,
              mm_token_type_ids=mm_token_type_ids,
          )
      except Exception:
        pass

    if isinstance(raw_input, str):
      tokens = self.encode(raw_input)
    elif isinstance(raw_input, list):
      tokens = raw_input
    else:
      raise ValueError(f"Unsupported input type: {type(raw_input)}")

    raw_token_length = len(tokens)
    total_len = self._max_seq_len + 1
    pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
    if pad_token_id is None:
      pad_token_id = getattr(self._tokenizer, "eos_token_id", None)
    if pad_token_id is None:
      pad_token_id = 0

    padded_tokens = torch.full(
        (1, total_len), pad_token_id, dtype=torch.long, device=self._device
    )
    tokens_len = min(raw_token_length, total_len)
    padded_tokens[0, :tokens_len] = torch.tensor(
        tokens[:tokens_len], dtype=torch.long, device=self._device
    )

    return VLFormattedInput(padded_tokens, tokens_len)

  def _compile_model_once(self) -> None:
    """Compiles the model for the target device."""
    assert self._model is not None
    self._model = device_utils.torch_compile(self._model, self._device.type)

  def get_logits_and_targets(
      self, formatted_input: quality_benchmark_model.FormattedInput
  ) -> tuple[torch.Tensor, torch.Tensor]:
    import typing
    import torch.distributed as dist

    vl_input = typing.cast(VLFormattedInput, formatted_input)
    assert self._model is not None

    if dist.is_initialized():
      rank = dist.get_rank()

      # 1. Broadcast input (always present)
      vl_input.input = _broadcast_tensor(vl_input.input, rank, self._device)

      # 2. Check and broadcast vision fields
      if rank == 0:
        has_vision_val = 1 if vl_input.pixel_values is not None else 0
        has_vision = torch.tensor(
            [has_vision_val], dtype=torch.long, device=self._device
        )
      else:
        has_vision = torch.tensor([0], dtype=torch.long, device=self._device)

      dist.broadcast(has_vision, src=0)

      if has_vision.item() == 1:
        vl_input.pixel_values = _broadcast_tensor(
            vl_input.pixel_values, rank, self._device
        )
        vl_input.image_grid_thw = _broadcast_tensor(
            vl_input.image_grid_thw, rank, self._device
        )
        vl_input.mm_token_type_ids = _broadcast_tensor(
            vl_input.mm_token_type_ids, rank, self._device
        )
      else:
        vl_input.pixel_values = None
        vl_input.image_grid_thw = None
        vl_input.mm_token_type_ids = None

    input_ids = vl_input.input[:, :-1]
    kwargs = {"input_ids": input_ids}
    if vl_input.pixel_values is not None:
      kwargs["pixel_values"] = vl_input.pixel_values
    if vl_input.image_grid_thw is not None:
      kwargs["image_grid_thw"] = vl_input.image_grid_thw
    if vl_input.mm_token_type_ids is not None:
      kwargs["mm_token_type_ids"] = vl_input.mm_token_type_ids[:, :-1]

    outputs = self._model(**kwargs)
    pred_logits = outputs.logits[0].to(torch.float32)
    target = vl_input.input[0, 1:].to(self._device)

    return pred_logits, target


def _broadcast_tensor(
    tensor: torch.Tensor | None, rank: int, device: torch.device
) -> torch.Tensor:
  """Helper to generically broadcast a tensor's shape, dtype, and raw data from Rank 0."""
  import torch.distributed as dist

  # Dtype mapping shims
  DTYPE_MAP = {
      torch.long: 0,
      torch.bfloat16: 1,
      torch.float32: 2,
  }
  REV_DTYPE_MAP = {
      0: torch.long,
      1: torch.bfloat16,
      2: torch.float32,
  }

  # 1. Broadcast ndim and dtype
  if rank == 0:
    assert tensor is not None
    ndim = tensor.ndim
    dtype_val = DTYPE_MAP.get(tensor.dtype, 2)
    meta_tensor = torch.tensor(
        [ndim, dtype_val], dtype=torch.long, device=device
    )
  else:
    meta_tensor = torch.tensor([0, 0], dtype=torch.long, device=device)

  dist.broadcast(meta_tensor, src=0)
  ndim = meta_tensor[0].item()
  global_dtype = REV_DTYPE_MAP[meta_tensor[1].item()]

  # 2. Broadcast shape array
  if rank == 0:
    assert tensor is not None
    shape_tensor = torch.tensor(
        list(tensor.shape), dtype=torch.long, device=device
    )
  else:
    shape_tensor = torch.empty(ndim, dtype=torch.long, device=device)

  dist.broadcast(shape_tensor, src=0)
  global_shape = list(shape_tensor.cpu().numpy())

  # 3. Allocate destination tensor on other ranks and broadcast raw data
  if rank != 0:
    tensor = torch.empty(global_shape, dtype=global_dtype, device=device)

  dist.broadcast(tensor, src=0)
  return tensor
